from __future__ import annotations

import os
import shutil
import socket
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@contextmanager
def temporary_postgres(base_dir: Path | None = None):
    """Spin up a throwaway PostgreSQL instance for tests."""

    workdir = Path(tempfile.mkdtemp())
    data_dir = workdir / "pgdata"
    log_file = workdir / "postgresql.log"

    env = os.environ.copy()
    env["PATH"] = "/usr/lib/postgresql/16/bin:" + env.get("PATH", "")

    data_dir.mkdir(parents=True, exist_ok=True)
    shutil.chown(workdir, user="postgres", group="postgres")
    shutil.chown(data_dir, user="postgres", group="postgres")

    init = subprocess.run(
        [
            "runuser",
            "-u",
            "postgres",
            "--",
            "initdb",
            "-D",
            str(data_dir),
            "--username",
            "postgres",
            "--auth",
            "trust",
            "--nosync",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if init.returncode != 0:
        raise RuntimeError(f"initdb failed: {init.stderr}")

    port = _find_free_port()
    start = subprocess.run(
        [
            "runuser",
            "-u",
            "postgres",
            "--",
            "pg_ctl",
            "-D",
            str(data_dir),
            "-l",
            str(log_file),
            "-o",
            f"-p {port}",
            "-w",
            "start",
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    if start.returncode != 0:
        raise RuntimeError(f"pg_ctl start failed: {start.stderr}\nlog: {log_file.read_text()}" if log_file.exists() else start.stderr)

    try:
        yield f"postgresql+psycopg://postgres@127.0.0.1:{port}/postgres"
    finally:
        subprocess.run(
            ["runuser", "-u", "postgres", "--", "pg_ctl", "-D", str(data_dir), "-m", "immediate", "stop"],
            env=env,
            capture_output=True,
        )
        shutil.rmtree(workdir, ignore_errors=True)
