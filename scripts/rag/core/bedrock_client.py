"""Shared AWS CLI invocation helpers with timeout/retry."""

from __future__ import annotations

import subprocess
import time


def run_aws_cli(
    cmd: list[str],
    *,
    timeout_sec: int,
    retries: int,
    retry_backoff_sec: float,
) -> subprocess.CompletedProcess[str]:
    attempts = max(1, retries + 1)
    last: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, attempts + 1):
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            if attempt >= attempts:
                raise RuntimeError(
                    f"AWS CLI timed out after {timeout_sec}s: {' '.join(cmd[:4])} ..."
                ) from exc
            time.sleep(retry_backoff_sec * attempt)
            continue

        last = result
        if result.returncode == 0:
            return result
        if attempt < attempts:
            time.sleep(retry_backoff_sec * attempt)

    assert last is not None
    stderr = (last.stderr or "").strip()
    raise RuntimeError(stderr or "AWS CLI call failed.")
