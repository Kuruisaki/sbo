"""Process behavior fixture, never an aerodynamic data source."""
from pathlib import Path
import re
import subprocess
import sys
import time

mode = sys.argv[1]
session = sys.stdin.read()
panels = int(re.search(r"^N (\d+)$", session, re.MULTILINE).group(1))
if mode != "missing_identity":
    print("XFOIL Version " + ("5.50" if mode == "wrong_version" else "6.99"), flush=True)
print("Number of panel nodes", panels - 1 if mode == "wrong_panels" else panels, flush=True)
if mode == "crash":
    print("simulated crash", file=sys.stderr, flush=True)
    sys.exit(7)
if mode == "flood":
    sys.stdout.buffer.write(b"\xff" + b"x" * 1024 * 1024 + b"\n")
    sys.stdout.buffer.flush()
if mode in ("timeout_child", "orphan_child"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    Path("child.pid").write_text(str(child.pid), encoding="ascii")
    if mode == "timeout_child":
        time.sleep(30)
# No polar is generated: normal process exit alone must not imply valid physics.
