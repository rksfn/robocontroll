# Robot Control Hub

One local service for a **Waveshare RoArm-M3** (USB), a **WAVE ROVER** (Wi-Fi) and an
**overhead camera** (OAK-D or any USB webcam). It gives you:

- a browser dashboard for smooth real-time manual control (keyboard, Xbox controller, clicks)
- camera ↔ arm calibration, so a pixel in the camera image becomes an arm position
- safe high-level tasks: **pick**, **place**, **move above**, **home**
- a small HTTP API and Python client that a vision model (Qwen-VL or similar) can drive

```
 camera frame ──► Qwen-VL: "red block is at pixel (300,150)"
                        │
                        ▼
 POST /api/task {"task":"pick","pixel":[300,150]}
                        │  calibration: pixel ─► arm X/Y on the table
                        ▼
 task runner ─► reach check (firmware IK) ─► 40 Hz streaming ─► RoArm-M3
```

---

## 1. Hardware and first start

| Device | Connection | Setting in `config.json` |
| --- | --- | --- |
| RoArm-M3 | USB serial, arm on its own power supply | `arm_port`: `COM7` (Windows) or `/dev/ttyUSB0` (Linux) |
| WAVE ROVER | Wi-Fi HTTP | `rover_url` |
| OAK-D | USB 3 | `camera_source`: `"oak"` |
| Any USB webcam | USB | `camera_source`: `"webcam:0"` (index 0, 1, …) |

Only one program can use the arm's serial port or the OAK-D at a time, so close other tools first.

**Windows** (PowerShell):

```powershell
cd robot-control-system
.\Start-Robot-Hub.ps1          # creates .venv, installs requirements, opens the dashboard
```

**Linux / macOS**:

```bash
cd robot-control-system
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python -m robot_hub.server     # then open http://127.0.0.1:8765
```

(On Linux add yourself to the `dialout` group for serial access.)

The hub always starts with **STOP latched**. Check the arm shows *Holding* and the camera image
appears, then press **Enable controls**.

Run the tests (no hardware needed, the arm is simulated):

```bash
python -m unittest -v
```

---

## 2. Manual control (dashboard)

| Input | What it does |
| --- | --- |
| Keyboard | `W`/`S` forward/back · `A`/`D` left/right · `R`/`F` up/down · `T`/`G` pitch · `Z`/`C` roll · `Q`/`E` open/close · hold `Shift` precise · `[` `]` speed |
| Xbox controller | left stick move · right stick ↕ up/down, ↔ roll · LB/RB pitch · LT/RT open/close · X precise · Y Arm⇄Rover · B STOP |
| Top / side maps | click to glide there; blue = reachable **at the current height and wrist pitch** |
| Camera image | pick a mode (*Move above*, *Pick*, *Place*, *Calibrate*) and click |
| Target boxes | type X/Y/Z/pitch/grip, press **Go** or Enter |
| `Space` / `Esc` / B | STOP: arm holds where it is, rover stops, running task is cancelled |

---

## 3. Camera calibration (do this once per camera position)

1. Mount the camera looking down at the table. Don't move it afterwards: if it moves, recalibrate.
2. In the dashboard click **Calibrate** above the camera image.
3. Put a small mark on the table. Jog the arm with the **gripper pointing down** (pitch ≈ 90°) until the tip just touches the mark.
4. Click the mark in the camera image. The hub stores *pixel ↔ arm X/Y/Z*.
5. Repeat for **5–8 marks spread over the whole area the arm can reach** (at least 4).

The panel shows the fit error in mm (aim for under 5 mm). The average Z of the points becomes the
**table height**, and the arm will refuse to go below it. Calibration is saved in `calibration.json`,
which git ignores because it belongs to your setup. Use *Undo* or *Clear* to redo points.

---

## 4. Tasks and API (for AI agents and scripts)

Base URL `http://127.0.0.1:8765`. By default the hub only listens on the local machine. To let another computer call it, set `server_host` to `0.0.0.0`, but only on a network you trust, because the API has no login.

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/state` | everything: arm pose, reach, task, calibration, camera, safety |
| GET | `/api/frame.jpg` | latest overhead frame (JPEG) |
| GET | `/api/pixel?u=..&v=..` | convert an image pixel to arm X/Y (mm) |
| POST | `/api/task` | run a task (below); returns immediately |
| GET | `/api/task` | task status: `running` / `done` / `failed` + message |
| POST | `/api/task/cancel` | cancel the running task |
| POST | `/api/stop` | latch STOP (any client may stop; only the dashboard can re-enable) |
| POST | `/api/action` | rover: `{"action":"drive","linear":-1..1,"turn":-1..1,"duration_ms":50..500}` |

Tasks (`POST /api/task`, add `"source":"qwen"` or similar):

```json
{"task": "pick",       "pixel": [312, 188]}
{"task": "place",      "pixel": [480, 240]}
{"task": "move_above", "pixel": [312, 188], "height_mm": 90}
{"task": "pick",       "xy": [230, 20]}
{"task": "move", "x": 250, "y": 0, "z": 100, "pitch": 45}
{"task": "gripper", "deg": 60}
{"task": "home"}
```

- `pixel` is in the camera image (`/api/state` → `camera.size`). If you use a resized image, also send `"image_size": [w, h]`.
- `xy` is in arm millimetres (x forward, y left, origin at the base axis). Use this if you compute positions yourself, e.g. from OAK-D depth.
- **Pick**: open → move above (hover) → descend to `grasp_height_mm` above the table → close → lift.
  It points the gripper straight down, or tilts it (75°, 60°, 45°) if the object is too far away for straight down.
- Every waypoint is checked **before** the task starts. An out-of-reach request fails at once with a
  clear message (e.g. *"At this height and pitch the arm can reach 72-352 mm"*) and the arm doesn't move.
- Tasks need the dashboard to be **enabled** by a human. STOP cancels them.

Python client (blocks until each task finishes, raises `RobotError` on failure):

```python
from robot_client import Robot
bot = Robot()                         # or Robot("http://<hub-ip>:8765")
bot.pick(pixel=(300, 150))
bot.place(pixel=(480, 240))
bot.home()
```

---

## 5. Qwen-VL (or any OpenAI-compatible vision model)

`qwen_agent.py` sends the overhead frame to the model, gets back **pixel points**, draws them on a
preview image, and (optionally) runs pick/place tasks.

```bash
# model server: Ollama shown; vLLM, LM Studio, llama.cpp server all work
ollama pull qwen2.5vl:7b
export MODEL_BASE_URL=http://127.0.0.1:11434/v1  VISION_MODEL=qwen2.5vl:7b

python qwen_agent.py --locate "red block"                              # find it, save last_plan.jpg
python qwen_agent.py --goal "put the red block in the bowl"            # plan only + preview
python qwen_agent.py --goal "put the red block in the bowl" --execute  # asks, then runs it
```

(PowerShell: `$env:MODEL_BASE_URL = '...'`, or `.\Run-Qwen-Observe.ps1 -Goal "..." [-Execute]`.)

**Coordinates.** Qwen2.5-VL answers in image pixels, Qwen3-VL in 0–1000 normalized units.
`--coords auto` chooses by model name; override with `--coords pixel|norm1000`. Always run
`--locate` first and check that the marker in `last_plan.jpg` sits on the object before using `--execute`.

The model can only choose `pick`, `place`, `move_above`, `home` or `done`. Its plan is checked
(e.g. no place while empty, points inside the image, max 6 steps), and each step goes through the
same reach checks as a human request.

---

## 6. Why the arm no longer "shoots away"

The firmware's direct Cartesian command (`T:1041`) runs inverse kinematics and writes the result to
the servos **without checking that it succeeded**. An unreachable target produces invalid joint
angles and the arm jumps. The hub now contains the same IK as the firmware (`robot_hub/kinematics.py`,
verified against the firmware's forward kinematics) plus the firmware's joint ranges, and:

- never sends a pose that fails IK or lies within 3° of a joint limit
- stops at the edge of reach when jogging, and clears a far target to the nearest reachable point
- moves long distances along an arc around the base, speeding up and slowing down smoothly
- if the real arm stops following (blocked or overloaded), holds where it is and reports it

Real reach depends strongly on height and wrist pitch. For example, with the gripper pointing down at
table height the arm reaches about 70–350 mm, but with the gripper level it only reaches 415–515 mm.
The blue areas on the dashboard maps show this live.

**Frame reminder:** z = 0 is the shoulder joint, about 126 mm above the bottom of the base, so a
table the arm stands on is at about z = −126.

---

## 7. Configuration reference (`config.json`)

| Key | Meaning |
| --- | --- |
| `arm_max_speed_mm_s`, `arm_goal_speed_mm_s`, `arm_accel_mm_s2` | jog / move speed and smoothness |
| `arm_z_min_mm`, `arm_z_max_mm`, `arm_reach_min_mm`, `arm_reach_max_mm` | extra user limits on top of the real reach check |
| `home_pose` | `[x, y, z, pitch, roll, grip]` for *Home* |
| `hover_height_mm`, `grasp_height_mm`, `release_height_mm` | pick/place heights above the table |
| `gripper_open_deg`, `gripper_closed_deg` | gripper angles for pick/place |
| `grasp_pitches_deg` | approach angles tried in order (90 = straight down) |
| `camera_source`, `camera_width`, `camera_height` | `"oak"` or `"webcam:N"`, and resolution |
| `calibration_file` | where to keep calibration (default `calibration.json`) |

## Files

```
robot_hub/arm.py         real-time arm engine (40 Hz stream, jog, goals, reach guard)
robot_hub/kinematics.py  RoArm-M3 IK/FK ported from Waveshare firmware + joint limits
robot_hub/vision.py      camera <-> arm calibration (homography, table height)
robot_hub/tasks.py       pick / place / move-above task runner
robot_hub/camera.py      OAK-D and USB webcam capture
robot_hub/rover.py       WAVE ROVER driver with watchdog
robot_hub/server.py      HTTP API + dashboard
static/dashboard.html    the dashboard
robot_client.py          Python client for scripts and agents
qwen_agent.py            vision-model planner (locate / plan / execute)
tests/                   simulated-hardware tests
```
