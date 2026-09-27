"""Vision-language model (Qwen2.5-VL / Qwen3-VL / similar) for the robot.

Talks to any OpenAI-compatible server (vLLM, Ollama, LM Studio ...), turns
"what / where" questions about the overhead camera frame into pixel
positions, and plans pick/place steps.  It never moves the robot itself:
the caller submits the resulting steps as normal, reach-checked tasks.

Accuracy notes
* Qwen2.5-VL answers in absolute pixels of the image it *received*, after
  resizing to multiples of 28 px.  We resize to exactly that size ourselves
  and scale the answer back, so there is no hidden offset.
* Qwen3-VL answers in 0-1000 normalized coordinates.
* The model is asked for "bbox_2d" boxes (its native grounding format) and
  the grasp point is the box centre.
"""

import base64
import json
import re
import time
import urllib.error
import urllib.request

POINT_ACTIONS = {"pick", "place", "move_above"}
PLAN_ACTIONS = POINT_ACTIONS | {"home", "done"}
MAX_STEPS = 6


def extract_json(text):
    """First JSON object or array in a model reply (tolerates ```json fences)."""
    text = re.sub(r"```(?:json)?", "", text)
    starts = [i for i in (text.find("{"), text.find("[")) if i >= 0]
    if not starts:
        raise ValueError("Model did not return JSON: " + text[:200])
    start = min(starts)
    closer = "}" if text[start] == "{" else "]"
    end = text.rfind(closer)
    if end <= start:
        raise ValueError("Model returned broken JSON: " + text[:200])
    return json.loads(text[start:end + 1])


class VisionModel:
    def __init__(self, base_url="http://127.0.0.1:8000/v1", model="", api_key="",
                 coords="auto", timeout=120):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.pinned = bool(model)          # a model named in config.json is kept; otherwise follow the server
        self.api_key = api_key
        self.coords = coords
        self.timeout = timeout

    # ------------------------------------------------------------ transport
    def _http(self, path, payload=None, timeout=None):
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(self.base_url + path, data, headers,
                                     method="GET" if data is None else "POST")
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read())

    def resolve_model(self):
        if not self.model:
            models = self._http("/models", timeout=5).get("data", [])
            if not models:
                raise RuntimeError("The model server has no models loaded")
            self.model = models[0]["id"]
        return self.model

    def status(self):
        now = time.time()
        cached = getattr(self, "_status_cache", None)
        if cached and cached[0] == self.base_url and now - cached[1] < 4:
            return cached[2]
        result = self._status()
        self._status_cache = (self.base_url, now, result)
        return result

    def _status(self):
        try:
            if not getattr(self, "pinned", False):
                # the server may have switched models (e.g. tunnel moved to a bigger GPU): follow it
                models = self._http("/models", timeout=5).get("data", [])
                if models and models[0]["id"] != self.model:
                    self.model = models[0]["id"]
            model = self.resolve_model()
            return {"ok": True, "model": model, "base_url": self.base_url,
                    "coords": self.mode()}
        except Exception as exc:
            return {"ok": False, "model": self.model, "base_url": self.base_url,
                    "error": f"{type(exc).__name__}: {exc}"}

    def mode(self):
        if self.coords != "auto":
            return self.coords
        return "norm1000" if re.search(r"qwen3|qwen-3", self.model or "", re.I) else "pixel"

    # ------------------------------------------------------------ image
    def prepare(self, jpeg, size, max_width=None):
        """-> (jpeg to send, (w, h) the model sees, scale back to camera px)."""
        w, h = size
        if self.mode() != "pixel":
            return jpeg, (w, h), (1.0, 1.0)
        f = min(1.0, float(max_width) / w) if max_width else 1.0
        nw, nh = max(28, round(w * f / 28) * 28), max(28, round(h * f / 28) * 28)
        if (nw, nh) == (w, h):
            return jpeg, (w, h), (1.0, 1.0)
        try:
            import cv2
            import numpy as np
            img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
            img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)
            jpeg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
        except ImportError:
            return jpeg, (w, h), (1.0, 1.0)
        return jpeg, (nw, nh), (w / nw, h / nh)

    def coord_text(self, seen):
        if self.mode() == "norm1000":
            return ("Coordinates are normalized 0-1000 "
                    "(0,0 = top-left, 1000,1000 = bottom-right).")
        return f"The image is {seen[0]}x{seen[1]} pixels; coordinates are pixels (0,0 = top-left)."

    def to_camera(self, box, seen, scale, size):
        """bbox_2d [x1,y1,x2,y2] or point [x,y] (model space) -> camera box + centre."""
        if not isinstance(box, (list, tuple)) or len(box) not in (2, 4):
            raise ValueError(f"Bad coordinates from model: {box!r}")
        v = [float(t) for t in box]
        if len(v) == 2:
            v = v + v
        if self.mode() == "norm1000":
            v = [v[0] * seen[0] / 1000, v[1] * seen[1] / 1000, v[2] * seen[0] / 1000, v[3] * seen[1] / 1000]
        v = [v[0] * scale[0], v[1] * scale[1], v[2] * scale[0], v[3] * scale[1]]
        cx, cy = (v[0] + v[2]) / 2, (v[1] + v[3]) / 2
        w, h = size
        if not (0 <= cx < w and 0 <= cy < h):
            raise ValueError(f"Model point {box} is outside the image")
        return [round(t, 1) for t in v], (round(cx, 1), round(cy, 1))

    def ask(self, prompt, jpeg, size, max_tokens=700, max_width=None):
        model = self.resolve_model()
        send, seen, scale = self.prepare(jpeg, size, max_width)
        payload = {"model": model, "temperature": 0.0, "max_tokens": max_tokens,
                   "messages": [{"role": "user", "content": [
                       {"type": "image_url", "image_url": {
                           "url": "data:image/jpeg;base64," + base64.b64encode(send).decode()}},
                       {"type": "text", "text": prompt.replace("{COORDS}", self.coord_text(seen))}]}]}
        try:
            reply = self._http("/chat/completions", payload)["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            if exc.code not in (400, 404) or self.pinned:
                raise
            self.model = ""
            payload["model"] = self.resolve_model()
            reply = self._http("/chat/completions", payload)["choices"][0]["message"]["content"]
        return reply, seen, scale

    # ------------------------------------------------------------ tasks
    def describe(self, jpeg, size, question):
        """Free-form question about the current camera frame -> plain text."""
        prompt = ("This image is from the camera of a robot (a rover and a robot arm). "
                  "Answer briefly and concretely. " + question)
        raw, _, _ = self.ask(prompt, jpeg, size, max_tokens=400)
        return {"answer": raw.strip(), "model": self.model}

    def locate(self, jpeg, size, what):
        """All matching objects: [{"label", "bbox", "pixel"}] (camera pixels)."""
        prompt = (f"Overhead camera view of a table next to a robot arm. Locate: {what}. "
                  "{COORDS} Output a JSON list only, one entry per matching object: "
                  '[{"bbox_2d": [x1, y1, x2, y2], "label": "<what it is, 1-3 words>"}]. '
                  "Output [] if nothing matches.")
        raw, seen, scale = self.ask(prompt, jpeg, size)
        data = extract_json(raw)
        if isinstance(data, dict):
            data = data.get("objects") or ([data] if "bbox_2d" in data else [])
        found = []
        for item in data[:20]:
            try:
                box, px = self.to_camera(item.get("bbox_2d") or item.get("point"), seen, scale, size)
            except (ValueError, AttributeError, TypeError):
                continue
            found.append({"label": str(item.get("label", what))[:40], "bbox": box, "pixel": px})
        return {"objects": found, "raw": raw, "model": self.model}

    def plan(self, jpeg, size, goal, holding=False):
        goal = re.sub(r"[-_]+", " ", str(goal)).strip()       # "pick-up-red-box" -> "pick up red box"
        prompt = f"""You control a robot arm. This photo is from the robot's camera looking down at the work area.
Think about which objects fit the goal (e.g. which items are trash, where a bin or drop spot is).
Goal: {goal}
The gripper is currently {"HOLDING an object" if holding else "empty"}.
{{COORDS}}
Reply with JSON only:
{{"observation": "max 12 words",
  "steps": [
    {{"action": "pick", "object": "name", "bbox_2d": [x1, y1, x2, y2]}},
    {{"action": "place", "target": "where", "bbox_2d": [x1, y1, x2, y2]}}
  ]}}
Actions: "pick" (grasp the object in the box), "place" (release the held object at the
centre of the box), "move_above" (hover over the box), "home", "done" (goal already met).
Rules: at most {MAX_STEPS} steps; pick only when the gripper is empty, place only when holding;
boxes must tightly enclose the real object or free target area you can SEE.
Be practical: if an object that reasonably matches the goal is visible (even if the wording is not
exact), you MUST include a "pick" step for it. If the goal only asks to pick something up, the plan
is just that one "pick" step. Return "steps": [] only if nothing suitable is visible at all."""
        raw, seen, scale = self.ask(prompt, jpeg, size, max_tokens=260)
        data = extract_json(raw)
        if not isinstance(data, dict):
            raise ValueError("Model plan was not a JSON object")
        steps, held = [], bool(holding)
        for step in (data.get("steps") or [])[:MAX_STEPS]:
            action = step.get("action")
            if action not in PLAN_ACTIONS:
                raise ValueError(f"Model proposed an unsupported action {action!r}")
            item = {"action": action, "label": str(step.get("object") or step.get("target") or "")[:40]}
            if action in POINT_ACTIONS:
                coords = step.get("bbox_2d") or step.get("bbox") or step.get("point")
                item["bbox"], item["pixel"] = self.to_camera(coords, seen, scale, size)
            if action == "pick":
                if held:
                    raise ValueError("Plan picks while the gripper is already holding something")
                held = True
            elif action == "place":
                if not held:
                    raise ValueError("Plan places without holding anything")
                held = False
            steps.append(item)
        return {"observation": str(data.get("observation", ""))[:300], "steps": steps,
                "raw": raw, "model": self.model}
