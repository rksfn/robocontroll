# robocontroll

Drive a Waveshare-style ESP32 rover from an Xbox controller on macOS or Windows.

## Controls

- Hold **A** and move the **left stick** to drive.
- Release **A** to stop.
- Press **Ctrl-C** to stop and exit.

Keep the rover's own control webpage closed while using this program.

## macOS with uv

```bash
uv run rover_control.py
```

## Windows with Python

```bat
py -m pip install -r requirements.txt
py rover_control.py
```

To build a standalone Windows executable on Windows:

```bat
py -m pip install pyinstaller
py -m PyInstaller --onefile --name rover-control rover_control.py
```

The executable is created at `dist\rover-control.exe`.

## Configuration

The defaults are rover URL `http://10.176.60.15` and half speed (`900` of `1800`). Override either with environment variables:

```bash
ROVER_URL=http://192.168.1.50 ROVER_MAX_SPEED=1200 uv run rover_control.py
```

In Windows PowerShell:

```powershell
$env:ROVER_URL = "http://192.168.1.50"
$env:ROVER_MAX_SPEED = "1200"
py rover_control.py
```

Run the checks without connecting to the rover:

```bash
uv run rover_control.py --self-test
```
