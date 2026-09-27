"""Command-line Qwen agent (the dashboard's "Ask Qwen" panel does the same).

  python qwen_agent.py --status
  python qwen_agent.py --locate "red block"
  python qwen_agent.py --goal "put the red block in the bowl"            # plan + preview
  python qwen_agent.py --goal "put the red block in the bowl" --execute  # asks, then runs

The hub talks to the model; this script only asks the hub, so it works the same
whether the model runs locally or on a cloud GPU (tunnelled to the hub's machine).
Robot hub URL: --robot or ROBOT_URL (default http://127.0.0.1:8765).
"""

import argparse
import json
import os
import sys

from robot_client import Robot, RobotError


def preview(jpeg, items, path):
    """Save the frame with the model's boxes so a human can check them."""
    try:
        import cv2
        import numpy as np
    except ImportError:
        return None
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    colors = {"pick": (0, 200, 255), "place": (80, 220, 80), "move_above": (255, 180, 60)}
    for i, it in enumerate(items, 1):
        c = colors.get(it.get("action"), (255, 255, 255))
        x1, y1, x2, y2 = (int(v) for v in it["bbox"])
        cv2.rectangle(img, (x1, y1), (x2, y2), c, 2)
        u, v = (int(t) for t in it["pixel"])
        cv2.drawMarker(img, (u, v), c, cv2.MARKER_CROSS, 16, 2)
        text = f"{i} {it.get('action', '')} {it.get('label', '')}".strip()
        cv2.putText(img, text[:40], (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, .5, c, 1, cv2.LINE_AA)
    cv2.imwrite(path, img)
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--status", action="store_true", help="check the model connection")
    ap.add_argument("--locate", help="find objects, e.g. 'red block'")
    ap.add_argument("--goal", help="goal in plain English")
    ap.add_argument("--execute", action="store_true", help="run the plan on the robot")
    ap.add_argument("--yes", action="store_true", help="do not ask before executing")
    ap.add_argument("--robot", default=os.getenv("ROBOT_URL", "http://127.0.0.1:8765"))
    ap.add_argument("--save", default="last_plan.jpg")
    args = ap.parse_args()
    bot = Robot(args.robot, source="qwen", timeout=90)

    try:
        if args.status or not (args.locate or args.goal):
            print(json.dumps(bot._req("/api/ai/status"), indent=2))
            return
        jpeg = bot.frame()
        if args.locate:
            res = bot._req("/api/ai/locate", {"what": args.locate})
            for o in res["objects"]:
                print(f"{o['label']:<20} pixel {o['pixel']}  arm X/Y {o.get('xy', '(calibrate first)')}")
            if not res["objects"]:
                print("Nothing found.")
            saved = preview(jpeg, res["objects"], args.save)
        else:
            res = bot._req("/api/ai/plan", {"goal": args.goal})
            print("Observation:", res["observation"])
            for i, s in enumerate(res["steps"], 1):
                print(f"  {i}. {s['action']:<10} {s['label']:<20} {s.get('pixel', '')} {s.get('xy', '')}")
            if not res["steps"]:
                print("No steps proposed.")
                return
            saved = preview(jpeg, [s for s in res["steps"] if "bbox" in s], args.save)
        if saved:
            print("Preview saved:", saved)
        if not args.goal or not args.execute:
            return
        if not args.yes and input("Execute this plan? [y/N] ").strip().lower() != "y":
            print("Cancelled.")
            return
        for i, s in enumerate(res["steps"], 1):
            print(f"[{i}/{len(res['steps'])}] {s['action']} {s['label']}")
            if s["action"] == "done":
                break
            if s["action"] == "home":
                bot.home()
            else:
                bot.task(task=s["action"], pixel=list(s["pixel"]))
        print("Finished.")
    except RobotError as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
