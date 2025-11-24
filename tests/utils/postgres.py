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


def _require_binary(name: str) -> str:
    env_override = os.environ.get(name.upper())
    if env_override:
        return env_override
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(f"required PostgreSQL binary '{name}' not found on PATH")
    return path


@contextmanager
def temporary_postgres(base_dir: Path | None = None):
    """Spin up a throwaway PostgreSQL instance for tests."""

    initdb_bin = _require_binary("initdb")
    pg_ctl_bin = _require_binary("pg_ctl")
    _require_binary("postgres")

    workdir_root = Path(tempfile.mkdtemp()) if base_dir is None else Path(base_dir)
    workdir_root.mkdir(parents=True, exist_ok=True)
    workdir = workdir_root / "temp_pg"
    data_dir = workdir / "pgdata"
    log_file = workdir / "postgresql.log"
    data_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()

    init = subprocess.run(
        [
            initdb_bin,
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
        raise RuntimeError(f"initdb failed: {init.stderr.strip()}")

    port = _find_free_port()
    start = subprocess.run(
        [
            pg_ctl_bin,
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
        err_text = (
            f"{start.stderr.strip()}\nlog: {log_file.read_text()}"
            if log_file.exists()
            else start.stderr.strip()
        )
        raise RuntimeError(f"pg_ctl start failed: {err_text}")

    try:
        yield f"postgresql+psycopg://postgres@127.0.0.1:{port}/postgres"
    finally:
        subprocess.run(
            [pg_ctl_bin, "-D", str(data_dir), "-m", "immediate", "stop"],
            env=env,
            capture_output=True,
            text=True,
        )
        shutil.rmtree(workdir, ignore_errors=True)
