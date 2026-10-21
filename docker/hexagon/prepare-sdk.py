"""Install QAIC once at image build time; keep the SDK read-only at runtime."""

import subprocess
from pathlib import Path

sdk = Path("/opt/hexagon-sdk")
subprocess.run(["make", "-C", str(sdk / "ipc/fastrpc/qaic")], check=True)
setup = sdk / "setup_sdk_env.source"
text = setup.read_text()
for command in (
    "make -C $HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/ clean",
    "make -C $HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/",
):
    text = text.replace(command, ": # QAIC was installed when the image was built.")
setup.write_text(text)
