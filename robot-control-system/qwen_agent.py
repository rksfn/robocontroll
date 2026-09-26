"""Qwen (or any OpenAI-compatible vision model) -> robot tasks.

The model looks at the overhead camera frame and answers with *where* things
are (pixel points).  The hub turns those points into arm coordinates using
the camera calibration, and runs safe, pre-checked pick / place tasks.

Examples
  python qwen_agent.py --locate "red block"            # just find it, save preview
  python qwen_agent.py --goal "put the red block in the bowl"          # plan only
  python qwen_agent.py --goal "put the red block in the bowl" --execute

Model endpoint (Ollama, vLLM, LM Studio, llama.cpp server ...):
  MODEL_BASE_URL  default http://127.0.0.1:11434/v1
  VISION_MODEL    default qwen2.5vl:7b
  MODEL_API_KEY   optional
Coordinates: Qwen2.5-VL answers in image pixels, Qwen3-VL in 0-1000
normalized units.  --coords auto picks by model name; check with --locate
and the saved preview image before using --execute.
"""

import argparse
import base64
import json
import os
import re
import sys
import time
import urllib.request

from robot_client import Robot, RobotError

MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")
MODEL_API_KEY = os.getenv("MODEL_API_KEY", "")

POINT_ACTIONS = {"pick", "place", "move_above"}
ALLOWED = POINT_ACTIONS | {"home", "stop", "drive", "done"}
MAX_STEPS = 6


# --------------------------------------------------------------- model I/O
def call_model(prompt, jpeg, model=VISION_MODEL, base_url=MODEL_BASE_URL):
    payload = {
        "model": model, "temperature": 0.1,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,"
                                                + base64.b64encode(jpeg).decode()}},
            {"type": "text", "text": prompt}]}],
    }
    headers = {"Content-Type": "application/json"}
    if MODEL_API_KEY:
        headers["Authorization"] = "Bearer " + MODEL_API_KEY
    req = urllib.request.Request(base_url + "/chat/completions", json.dumps(payload).encode(),
                                 headers, method="POST")
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"]


def extract_json(text):
    text = re.sub(r"```(?:json)?", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model did not return JSON: " + text[:200])
    return json.loads(match.group(0))


def coord_mode(model, requested):
    if requested != "auto":
        return requested
    return "norm1000" if re.search(r"qwen3|qwen-3", model, re.I) else "pixel"


def to_pixel(point, size, mode):
    """point [x, y] or bbox [x1, y1, x2, y2] -> pixel (u, v) in the camera frame."""
    if not isinstance(point, (list, tuple)) or len(point) not in (2, 4):
        raise ValueError(f"Bad point {point!r}")
    pt = [float(v) for v in point]
    if len(pt) == 4:
        pt = [(pt[0] + pt[2]) / 2, (pt[1] + pt[3]) / 2]
    w, h = size
    if mode == "norm1000":
        pt = [pt[0] * w / 1000, pt[1] * h / 1000]
    u, v = pt
    if not (0 <= u < w and 0 <= v < h):
        raise ValueError(f"Point {point} is outside the {w}x{h} image")
    return round(u, 1), round(v, 1)


def coord_text(size, mode):
    w, h = size
    if mode == "norm1000":
        return ("Give every point as [x, y] in normalized coordinates from 0 to 1000 "
                "(0,0 = top-left, 1000,1000 = bottom-right).")
    return (f"The image is {w}x{h} pixels. Give every point as [x, y] in pixels "
            "(0,0 = top-left).")


# --------------------------------------------------------------- tasks
def locate(bot, jpeg, size, what, mode, model):
    prompt = (f"This is an overhead camera view of a table with a robot arm. "
              f"Find: {what}. {coord_text(size, mode)} Return JSON only: "
              '{"found": true, "label": "...", "point": [x, y], "bbox": [x1, y1, x2, y2]} '
              'or {"found": false, "reason": "..."}. The point must be the centre of the '
              "object where a gripper coming from above should grasp it.")
    raw = call_model(prompt, jpeg, model)
    data = extract_json(raw)
    if not data.get("found"):
        return None, data, raw
    return to_pixel(data.get("point") or data.get("bbox"), size, mode), data, raw


def plan(bot, jpeg, size, goal, mode, model):
    st = bot.state()
    holding = st["task"].get("holding")
    prompt = f"""You control a robot arm through an OVERHEAD camera image of a table.
Goal: {goal}
The gripper is currently {"HOLDING an object" if holding else "empty"}.
{coord_text(size, mode)}
Return JSON only, no prose:
{{"observation": "short description of relevant objects",
  "steps": [
    {{"action": "pick", "object": "name", "point": [x, y]}},
    {{"action": "place", "target": "where", "point": [x, y]}}
  ]}}
Allowed actions: "pick" (grasp object centred at point), "place" (release held object at point),
"move_above" (hover over point), "home" (return to rest), "done" (goal already achieved).
Rules: at most {MAX_STEPS} steps; only pick when the gripper is empty and place only when holding;
points must be ON visible objects/free table space; if the goal is impossible or the object
is not visible, return {{"observation": "...", "steps": []}}."""
    raw = call_model(prompt, jpeg, model)
    data = extract_json(raw)
    steps = []
    held = bool(holding)
    for step in data.get("steps", [])[:MAX_STEPS]:
        action = step.get("action")
        if action not in ALLOWED - {"stop", "drive"}:
            raise ValueError(f"Model returned unsupported action {action!r}")
        item = {"action": action, "label": step.get("object") or step.get("target") or ""}
        if action in POINT_ACTIONS:
            item["pixel"] = to_pixel(step.get("point") or step.get("bbox"), size, mode)
        if action == "pick":
            if held:
                raise ValueError("Plan picks while already holding something")
            held = True
        if action == "place":
            if not held:
                raise ValueError("Plan places without holding anything")
            held = False
        steps.append(item)
    return data.get("observation", ""), steps, raw


def preview(jpeg, marks, path):
    """Save the frame with numbered markers so a human can check the plan."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    colors = {"pick": (0, 200, 255), "place": (80, 220, 80), "move_above": (255, 180, 60)}
    for i, (action, (u, v), label) in enumerate(marks, 1):
        c = colors.get(action, (255, 255, 255))
        cv2.drawMarker(img, (int(u), int(v)), c, cv2.MARKER_CROSS, 22, 2)
        cv2.circle(img, (int(u), int(v)), 14, c, 2)
        cv2.putText(img, f"{i} {action} {label}"[:40], (int(u) + 16, int(v) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, c, 1, cv2.LINE_AA)
    cv2.imwrite(path, img)
    return path


def execute(bot, steps):
    for i, step in enumerate(steps, 1):
        action = step["action"]
        print(f"[{i}/{len(steps)}] {action} {step.get('label', '')} {step.get('pixel', '')}")
        if action == "done":
            break
        if action == "home":
            bot.home()
        elif action == "pick":
            bot.pick(pixel=step["pixel"])
        elif action == "place":
            bot.place(pixel=step["pixel"])
        elif action == "move_above":
            bot.move_above(pixel=step["pixel"])
    print("Finished.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--goal", help="natural-language goal, e.g. 'put the red block in the bowl'")
    ap.add_argument("--locate", help="only find this object and print/save its point")
    ap.add_argument("--execute", action="store_true", help="run the plan on the robot")
    ap.add_argument("--yes", action="store_true", help="do not ask before executing")
    ap.add_argument("--coords", default="auto", choices=["auto", "pixel", "norm1000"])
    ap.add_argument("--model", default=VISION_MODEL)
    ap.add_argument("--robot", default=os.getenv("ROBOT_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--save", default="last_plan.jpg", help="annotated preview image path")
    args = ap.parse_args()
    if not args.goal and not args.locate:
        ap.error("give --goal or --locate")

    bot = Robot(args.robot, source="qwen")
    st = bot.state()
    size = st["camera"]["size"]
    if not st["camera"]["connected"] or not size:
        sys.exit("Camera not available: " + str(st["camera"].get("error")))
    if not st["calibration"]["ready"]:
        print("WARNING: camera not calibrated - points cannot be turned into arm moves yet.")
    jpeg = bot.frame()
    mode = coord_mode(args.model, args.coords)
    print(f"Model {args.model}, coordinates: {mode}, image {size[0]}x{size[1]}")

    if args.locate:
        px, data, raw = locate(bot, jpeg, size, args.locate, mode, args.model)
        print(json.dumps(data, indent=2))
        if px:
            info = bot.pixel_to_arm(*px) if st["calibration"]["ready"] else None
            print(f"Pixel {px}" + (f" -> arm X/Y {info} mm" if info else ""))
            saved = preview(jpeg, [("pick", px, args.locate)], args.save)
            if saved:
                print("Preview saved:", saved)
        return

    observation, steps, raw = plan(bot, jpeg, size, args.goal, mode, args.model)
    print("Observation:", observation)
    if not steps:
        print("No steps proposed (object not visible or goal impossible).")
        return
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s['action']:<10} {s['label']:<20} {s.get('pixel', '')}")
    saved = preview(jpeg, [(s["action"], s["pixel"], s["label"]) for s in steps if "pixel" in s], args.save)
    if saved:
        print("Preview saved:", saved, "- check the markers are on the right objects.")
    if not args.execute:
        print("Plan only. Add --execute to run it.")
        return
    if not args.yes and input("Execute this plan? [y/N] ").strip().lower() != "y":
        print("Cancelled.")
        return
    try:
        execute(bot, steps)
    except RobotError as exc:
        sys.exit(f"Stopped: {exc}")


if __name__ == "__main__":
    main()
