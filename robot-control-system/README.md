# Robot Control System

One local service combines the WAVE ROVER, RoArm M3 and OAK-D camera. It provides
a browser dashboard for manual operation and a small validated API for Qwen or
another vision model.

## Connections

- Rover: Wi-Fi HTTP at `http://10.176.60.15`
- Arm: USB serial on `COM7`, with the arm's normal power supply
- Camera: OAK-D connected directly to a USB 3 port
- Dashboard: local laptop page at `http://127.0.0.1:8765`

Edit `config.json` if an address or COM port changes. The server binds to
`127.0.0.1`, so another computer cannot issue commands over the network.

## Start

Close the old rover/arm controller and camera examples first, because only one
program can use COM7 or the OAK-D at a time. Then run:

```powershell
cd "C:\Users\peter\Documents\ChatGPT\New project\robot-control-system"
.\Start-Robot-Hub.ps1
```

The dashboard opens automatically. It always starts with STOP latched. Check the
camera and device status, then click **Enable manual and agent actions**.
If the OAK-D is connected after startup, the camera service retries automatically.

For visible diagnostic output:

```powershell
.\Run-Robot-Hub-Debug.ps1
```

## Manual controls

The dashboard shows the OAK-D stream, rover hold-to-move buttons, held XYZ arm
movement, absolute XYZ target fields, gripper controls, device status and the
action log. Space or Escape latches STOP. Rover commands expire after 300 ms.
The local operator can move to an XYZ target up to 75 mm from the current pose;
wrist pitch, roll and grip remain fixed. AI requests remain limited to small
10 mm nudges and cannot use the absolute-target action.

## Agent API

Read current state:

```text
GET http://127.0.0.1:8765/api/state
```

Read a current camera frame:

```text
GET http://127.0.0.1:8765/api/frame.jpg
```

Submit one action with `POST /api/action`:

```json
{"source":"qwen","action":"drive","linear":0.2,"turn":0,"duration_ms":200,"request_id":"step-1"}
```

```json
{"source":"qwen","action":"arm_nudge","dx_mm":5,"dy_mm":0,"dz_mm":0,"gripper_deg":0,"request_id":"step-2"}
```

Allowed limits are enforced again inside the service:

- Drive values: `-1` to `1`, duration 50 to 500 ms
- Arm XYZ: maximum 10 mm per action
- Gripper: maximum 5 degrees per action
- The model can stop but cannot clear the STOP latch

## Qwen or another vision model

`qwen_agent.py` uses an OpenAI-compatible local vision endpoint. The default is
an Ollama-style endpoint at `http://127.0.0.1:11434/v1` with model
`qwen2.5vl:7b`. Change these without editing code:

```powershell
$env:MODEL_BASE_URL = 'http://127.0.0.1:11434/v1'
$env:VISION_MODEL = 'YOUR_VISION_MODEL_NAME'
.\Run-Qwen-Observe.ps1 -Goal 'Describe the scene and identify any bottle'
```

This defaults to observe-only and prints the proposed JSON action. To execute a
validated action, run Python directly with `--execute`. The dashboard must first
be enabled by a human:

```powershell
.\.venv\Scripts\python.exe qwen_agent.py --goal "Move slowly toward the bottle" --execute
```

Do not start with an unattended loop. First verify individual proposed actions,
camera geometry, rover direction and arm coordinate direction.

## Spatial detection

The default camera mode is `rgb` because it is the lowest-bandwidth reliable
stream. To add YOLO labels and OAK-D spatial X/Y/Z measurements, change
`camera_mode` in `config.json` to `spatial`. This loads `yolov6-nano` on the
camera pipeline and may require USB 3 (`UsbSpeed.SUPER`). Switch back to `rgb` if
the device negotiates USB 2 or the spatial pipeline cannot start.

## Tests

Tests use fake hardware and do not move the robot:

```powershell
.\.venv\Scripts\python.exe -m unittest -v
```
