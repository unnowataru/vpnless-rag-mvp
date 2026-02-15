"""Shared AWS CLI invocation helpers with timeout/retry."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Protocol


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


class BedrockClient(Protocol):
    def rerank(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        ...

    def converse(self, *, model_id: str, payload: dict[str, Any]) -> str:
        ...


AwsCliRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class AwsCliBedrockClient(BedrockClient):
    region: str
    profile: str
    timeout_sec: int
    retries: int
    retry_backoff_sec: float
    runner: AwsCliRunner = run_aws_cli

    def rerank(self, *, payload: dict[str, Any]) -> dict[str, Any]:
        cmd = [
            "aws",
            "bedrock-agent-runtime",
            "rerank",
            "--region",
            self.region,
            "--profile",
            self.profile,
            "--cli-input-json",
            self._write_payload_file(payload),
            "--no-cli-pager",
            "--output",
            "json",
        ]
        result = self._run(cmd)
        if not result.stdout.strip():
            return {}
        try:
            return json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError("Failed to parse Bedrock rerank response as JSON.") from exc

    def converse(self, *, model_id: str, payload: dict[str, Any]) -> str:
        cmd = [
            "aws",
            "bedrock-runtime",
            "converse",
            "--region",
            self.region,
            "--profile",
            self.profile,
            "--model-id",
            model_id,
            "--cli-input-json",
            self._write_payload_file(payload),
            "--no-cli-pager",
            "--query",
            "output.message.content[0].text",
            "--output",
            "text",
        ]
        result = self._run(cmd)
        return result.stdout.strip()

    def _run(self, cmd: list[str]) -> subprocess.CompletedProcess[str]:
        payload_arg = ""
        try:
            payload_arg = cmd[cmd.index("--cli-input-json") + 1]
            return self.runner(
                cmd,
                timeout_sec=self.timeout_sec,
                retries=self.retries,
                retry_backoff_sec=self.retry_backoff_sec,
            )
        finally:
            if payload_arg.startswith("file://"):
                payload_path = payload_arg.replace("file://", "", 1)
                try:
                    os.remove(payload_path)
                except OSError:
                    pass

    @staticmethod
    def _write_payload_file(payload: dict[str, Any]) -> str:
        with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".json", delete=False) as tmp:
            json.dump(payload, tmp, ensure_ascii=False)
            return f"file://{tmp.name}"
