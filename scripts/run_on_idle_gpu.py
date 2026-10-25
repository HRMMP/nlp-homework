"""Run one command on a rechecked idle GPU, preserving logs and exit status.

Advisory locks coordinate this user's launches, not other users or a scheduler.
"""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import subprocess
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path


def idle_candidates(gpus, occupied, max_used_mib=256, max_utilization=5):
    return sorted(
        [
            gpu
            for gpu in gpus
            if gpu["uuid"] not in occupied
            and gpu["memory_used_mib"] <= max_used_mib
            and gpu["utilization_percent"] <= max_utilization
        ],
        key=lambda gpu: (gpu["memory_used_mib"], gpu["index"]),
    )


def snapshot():
    data = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.used,utilization.gpu,name",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    gpus = []
    for row in csv.reader(data.splitlines(), skipinitialspace=True):
        index, uuid, used, utilization, name = row
        gpus.append(
            {
                "index": int(index),
                "uuid": uuid,
                "name": name,
                "memory_used_mib": int(used),
                "utilization_percent": int(utilization),
            }
        )
    processes = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid",
            "--format=csv,noheader,nounits",
        ],
        text=True,
    )
    occupied = {
        row[0]
        for row in csv.reader(processes.splitlines(), skipinitialspace=True)
        if row and row[0].startswith("GPU-")
    }
    return gpus, occupied


def reserve_idle_gpu(max_used_mib, max_utilization):
    initial, occupied = snapshot()
    directory = Path(tempfile.gettempdir()) / f"nlp-homework-gpu-locks-{os.getuid()}"
    directory.mkdir(mode=0o700, exist_ok=True)
    for candidate in idle_candidates(initial, occupied, max_used_mib, max_utilization):
        handle = (directory / f"{candidate['uuid']}.lock").open("a")
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            handle.close()
            continue
        time.sleep(2)
        current, occupied = snapshot()
        eligible = idle_candidates(current, occupied, max_used_mib, max_utilization)
        selected = next(
            (gpu for gpu in eligible if gpu["uuid"] == candidate["uuid"]), None
        )
        if selected is not None:
            return selected, handle, initial, current
        handle.close()
    raise RuntimeError("No GPU passed both idle checks; no command was launched")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--max-used-mib", type=int, default=256)
    parser.add_argument("--max-utilization", type=int, default=5)
    parser.add_argument("--background", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide a command after --")
    if args.record.resolve() == args.log.resolve():
        parser.error("Record and log paths must differ")
    for path in (args.record, args.log):
        if path.exists():
            parser.error(f"Refusing to overwrite {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    if args.background:
        argv = sys.argv[1:]
        argv.remove("--background")
        process = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), *argv],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        print(
            json.dumps(
                {
                    "guardian_pid": process.pid,
                    "record": str(args.record),
                    "log": str(args.log),
                }
            )
        )
        return 0

    record = {
        "state": "selecting_gpu",
        "guardian_pid": os.getpid(),
        "command": command,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "idle_thresholds": {
            "max_used_mib": args.max_used_mib,
            "max_utilization": args.max_utilization,
        },
    }
    with args.record.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2)
    lock = None
    with args.log.open("x", encoding="utf-8") as log:
        try:
            gpu, lock, before, after = reserve_idle_gpu(
                args.max_used_mib, args.max_utilization
            )
            environment = dict(os.environ)
            environment["CUDA_VISIBLE_DEVICES"] = gpu["uuid"]
            for name, value in (
                ("OMP_NUM_THREADS", "4"),
                ("MKL_NUM_THREADS", "4"),
                ("OPENBLAS_NUM_THREADS", "1"),
            ):
                environment.setdefault(name, value)
            process = subprocess.Popen(
                command, env=environment, stdout=log, stderr=subprocess.STDOUT
            )
            record.update(
                {
                    "state": "running",
                    "pid": process.pid,
                    "gpu": gpu,
                    "gpu_snapshot_before": before,
                    "gpu_snapshot_recheck": after,
                    "cuda_visible_devices": gpu["uuid"],
                }
            )
            args.record.write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
            exit_code = process.wait()
            record.update(
                {
                    "state": "completed" if exit_code == 0 else "failed",
                    "exit_code": exit_code,
                }
            )
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as error:
            log.write(traceback.format_exc())
            record.update({"state": "failed", "exit_code": 1, "error": str(error)})
        finally:
            if lock is not None:
                lock.close()
            record["finished_utc"] = datetime.now(timezone.utc).isoformat()
            args.record.write_text(
                json.dumps(record, indent=2) + "\n", encoding="utf-8"
            )
    return record["exit_code"]


if __name__ == "__main__":
    raise SystemExit(main())
