"""Install QAIC once at image build time; keep the SDK read-only at runtime."""

import subprocess
from pathlib import Path


def prepare_sdk(sdk: Path) -> None:
    # CI packages may omit the helpers needed by both generated ARM hosts.
    for suffix in ("c", "h"):
        helper = sdk / f"utils/examples/dsp_capabilities_utils.{suffix}"
        if not helper.is_file():
            raise RuntimeError(f"Missing {helper}; supply the full SDK with its example helpers")

    qaic_dir = sdk / "ipc/fastrpc/qaic"
    qaic = qaic_dir / "bin/qaic"
    # Reuse CI packages' preinstalled QAIC even when they have no Makefile.
    if not qaic.is_file():
        subprocess.run(["make", "-C", str(qaic_dir)], check=True)
    # The SDK's make copies a 0750 binary as root. Runtime UID/GID is arbitrary,
    # so group ownership from COPY --chown cannot make this executable usable.
    qaic.chmod(0o755)
    setup = sdk / "setup_sdk_env.source"
    text = setup.read_text()
    for command in (
        "make -C $HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/ clean",
        "make -C $HEXAGON_SDK_ROOT/ipc/fastrpc/qaic/",
    ):
        text = text.replace(command, ": # QAIC was installed when the image was built.")
    setup.write_text(text)


if __name__ == "__main__":
    prepare_sdk(Path("/opt/hexagon-sdk"))
