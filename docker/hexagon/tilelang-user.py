"""Register the host UID in the disposable container, then drop privileges."""

import grp
import os
import pwd
import subprocess
import sys
from pathlib import Path

uid = int(os.environ.get("LOCAL_UID", "1000"))
gid = int(os.environ.get("LOCAL_GID", "1000"))
home = Path(f"/tmp/tilelang-home-{uid}")
try:
    grp.getgrgid(gid)
except KeyError:
    subprocess.run(["groupadd", "-g", str(gid), f"hostgid{gid}"], check=True)
try:
    pwd.getpwuid(uid)
except KeyError:
    subprocess.run(["useradd", "-M", "-u", str(uid), "-g", str(gid), "-d", str(home), "-s", "/bin/bash", f"hostuid{uid}"], check=True)
home.mkdir(parents=True, exist_ok=True)
os.chown(home, uid, gid)
os.environ["HOME"] = str(home)
os.environ["XDG_CACHE_HOME"] = str(home / ".cache")
os.environ["TILELANG_HEXAGON_SSH_KNOWN_HOSTS"] = os.environ.get("TILELANG_HEXAGON_SSH_KNOWN_HOSTS") or str(home / ".ssh/known_hosts")
# SDK files belong to group 1000; host files still use the caller's UID/GID.
os.setgroups([1000])
os.setgid(gid)
os.setuid(uid)
for directory in [home / ".cache", Path("/output/tmp")]:
    directory.mkdir(parents=True, exist_ok=True)
os.execvp(sys.argv[1], sys.argv[1:])
