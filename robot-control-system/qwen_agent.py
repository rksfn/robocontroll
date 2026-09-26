"""Vision-agent adapter for an OpenAI-compatible Qwen or similar local model."""
import argparse
import base64
import json
import os
import re
import time
import urllib.error
import urllib.request


ROBOT_URL = os.getenv("ROBOT_URL", "http://127.0.0.1:8765").rstrip("/")
MODEL_BASE_URL = os.getenv("MODEL_BASE_URL", "http://127.0.0.1:11434/v1").rstrip("/")
VISION_MODEL = os.getenv("VISION_MODEL", "qwen2.5vl:7b")

ALLOWED = {
    "stop": set(),
    "drive": {"linear", "turn", "duration_ms"},
    "arm_nudge": {"dx_mm", "dy_mm", "dz_mm", "gripper_deg"},
    "observe": {"description"},
}


def request(url, method="GET", data=None, timeout=20):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(url, data=body, method=method,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        raw = response.read()
        return raw if "image/" in response.headers.get("Content-Type", "") else json.loads(raw)


def parse_action(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("Model did not return a JSON object")
    action = json.loads(match.group(0))
    name = action.get("action")
    if name not in ALLOWED:
        raise ValueError("Model returned an unsupported action")
    extra = set(action).difference(ALLOWED[name] | {"action", "reason"})
    if extra:
        raise ValueError("Unexpected action fields: " + ", ".join(sorted(extra)))
    if name == "drive":
        if abs(float(action.get("linear", 0))) > 1 or abs(float(action.get("turn", 0))) > 1:
            raise ValueError("Drive values exceed limits")
        if not 50 <= int(action.get("duration_ms", 0)) <= 500:
            raise ValueError("Drive duration exceeds limits")
    if name == "arm_nudge":
        if any(abs(float(action.get(key, 0))) > 10 for key in ("dx_mm", "dy_mm", "dz_mm")):
            raise ValueError("Arm XYZ nudge exceeds limits")
        if abs(float(action.get("gripper_deg", 0))) > 5:
            raise ValueError("Gripper nudge exceeds limits")
    return action


def ask_model(goal):
    state = request(ROBOT_URL + "/api/state")
    frame = request(ROBOT_URL + "/api/frame.jpg")
    prompt = f"""You are the perception planner for a small rover with a RoArm M3.
Goal: {goal}
Current state: {json.dumps(state, separators=(',', ':'))}
Choose exactly one bounded next action. Return JSON only.
Allowed forms:
{{"action":"observe","description":"what you see","reason":"..."}}
{{"action":"stop","reason":"..."}}
{{"action":"drive","linear":-1..1,"turn":-1..1,"duration_ms":50..500,"reason":"..."}}
{{"action":"arm_nudge","dx_mm":-10..10,"dy_mm":-10..10,"dz_mm":-10..10,"gripper_deg":-5..5,"reason":"..."}}
Prefer observe or stop when uncertain. Never claim an object position that is not visible."""
    payload = {
        "model": VISION_MODEL,
        "temperature": 0.1,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url":
                "data:image/jpeg;base64," + base64.b64encode(frame).decode()}},
        ]}],
    }
    response = request(MODEL_BASE_URL + "/chat/completions", "POST", payload, timeout=120)
    text = response["choices"][0]["message"]["content"]
    return parse_action(text), text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--goal", required=True)
    parser.add_argument("--execute", action="store_true",
                        help="Submit the validated action; default is observe-only")
    args = parser.parse_args()
    action, raw = ask_model(args.goal)
    print(json.dumps(action, indent=2))
    if args.execute and action["action"] != "observe":
        action.update({"source": "qwen", "request_id": f"qwen-{int(time.time())}"})
        print(json.dumps(request(ROBOT_URL + "/api/action", "POST", action), indent=2))
    elif args.execute:
        print("Observation only; no robot action submitted.")
    else:
        print("Observe-only mode. Add --execute to submit a validated action.")


if __name__ == "__main__":
    main()
