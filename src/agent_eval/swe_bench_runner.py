from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .schema import BenchmarkCase, RunnerResult


class SWEBenchRunner:
    """Minimal repo clone/apply/test runner with cleanup by default."""

    def __init__(self, timeout_sec: int = 900, keep_worktree: bool = False):
        self.timeout_sec = timeout_sec
        self.keep_worktree = keep_worktree

    def run(self, case: BenchmarkCase, patch: str, test_command: str | None = None) -> RunnerResult:
        if not patch.strip() or not patch.lstrip().startswith("diff --git"):
            return RunnerResult(status="skipped", logs="No valid unified git patch was provided.")
        repo = case.metadata.get("repo")
        base_commit = case.metadata.get("base_commit")
        if not repo or not base_commit:
            return RunnerResult(status="skipped", logs="repo/base_commit missing; cannot run SWE-bench locally.")

        temp_root = Path(tempfile.mkdtemp(prefix="pcu-bench-"))
        worktree = temp_root / "repo"
        cleanup = not self.keep_worktree
        try:
            clone_url = repo if str(repo).startswith(("http://", "https://", "git@")) else f"https://github.com/{repo}.git"
            clone = self._run(["git", "clone", "--quiet", clone_url, str(worktree)], cwd=temp_root)
            if clone.returncode != 0:
                return RunnerResult(status="error", logs=clone.stdout + clone.stderr, worktree=str(worktree), cleanup_performed=False)
            checkout = self._run(["git", "checkout", "--quiet", str(base_commit)], cwd=worktree)
            if checkout.returncode != 0:
                return RunnerResult(status="error", logs=checkout.stdout + checkout.stderr, worktree=str(worktree), cleanup_performed=False)
            apply = self._run(["git", "apply", "--whitespace=nowarn", "-"], cwd=worktree, input_text=patch)
            if apply.returncode != 0:
                return RunnerResult(status="fail", logs=apply.stdout + apply.stderr, worktree=str(worktree), cleanup_performed=cleanup)
            diff = self._run(["git", "diff", "--"], cwd=worktree)
            command = test_command or case.evaluation.get("test_command") or "python -m pytest"
            test = self._run_shell(command, cwd=worktree)
            status = "pass" if test.returncode == 0 else "fail"
            return RunnerResult(
                status=status,
                logs=(test.stdout + test.stderr)[-20000:],
                diff=diff.stdout[-20000:],
                worktree=str(worktree) if self.keep_worktree else None,
                cleanup_performed=cleanup,
                metadata={"test_command": command},
            )
        except subprocess.TimeoutExpired as exc:
            return RunnerResult(status="error", logs=f"Timeout after {self.timeout_sec}s: {exc}", worktree=str(worktree), cleanup_performed=cleanup)
        finally:
            if cleanup:
                shutil.rmtree(temp_root, ignore_errors=True)

    def _run(self, args: list[str], cwd: Path, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            args,
            cwd=str(cwd),
            input=input_text,
            text=True,
            capture_output=True,
            timeout=self.timeout_sec,
        )

    def _run_shell(self, command: str, cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=self.timeout_sec,
            shell=True,
        )

