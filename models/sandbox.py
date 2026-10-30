"""Phase 15 external sandbox: the model's only source of ground truth.

Verification is **outsourced**: the model never decides whether code is correct,
it only routes a request to a tool. This module runs the tool and returns the
bytes the environment produced (stdout, stderr, exit status, file contents).

Hardening applied here:

* A private root directory; every relative path is resolved inside it and
  absolute paths, ``..`` escapes and symlinks that leave the root are rejected.
* ``python -I -B`` (isolated, no bytecode writes) with a scrubbed environment,
  a wall-clock timeout and POSIX resource limits (CPU, address space, file
  size, core dumps).
* Bounded stdout/stderr and bounded file reads/writes.

This is *defence in depth*, not a security boundary: it is a child process
running as the same user on the same host with network access. Do not run
untrusted multi-tenant workloads through it.
"""
from __future__ import annotations

import os
import resource
import subprocess
import sys
import tempfile
from pathlib import Path

DEFAULT_TIMEOUT: float = 5.0
MAX_OUTPUT_BYTES: int = 8192
MAX_FILE_BYTES: int = 65536
MEMORY_LIMIT_BYTES: int = 512 * 1024 * 1024
FILE_SIZE_LIMIT_BYTES: int = 1024 * 1024
STDERR_MARK: bytes = b"\n---STDERR---\n"
TOOL_NAMES: tuple[bytes, ...] = (b"exec_python", b"read_file", b"write_file")


class SandboxViolation(RuntimeError):
    """A request that escapes the sandbox or exceeds a configured bound."""


class ByteSandbox:
    """A directory-confined, time- and size-bounded execution environment."""

    def __init__(self, root: str | Path | None = None, *, timeout: float = DEFAULT_TIMEOUT,
                 max_output_bytes: int = MAX_OUTPUT_BYTES, max_file_bytes: int = MAX_FILE_BYTES,
                 python_executable: str | None = None, enabled: bool = True) -> None:
        if timeout <= 0 or not isinstance(timeout, (int, float)):
            raise ValueError("timeout must be a positive number of seconds")
        if max_output_bytes < 1 or max_file_bytes < 1:
            raise ValueError("sandbox byte limits must be positive")
        self.timeout = float(timeout)
        self.max_output_bytes = int(max_output_bytes)
        self.max_file_bytes = int(max_file_bytes)
        self.python_executable = python_executable or sys.executable
        self.enabled = bool(enabled)
        self._owns_root = root is None
        self.root = Path(tempfile.mkdtemp(prefix="bytecore_sandbox_")) if root is None else Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.root = self.root.resolve()

    def close(self) -> None:
        """Remove the sandbox directory only when this instance created it."""
        if self._owns_root and self.root.exists():
            for path in sorted(self.root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
                try:
                    path.unlink()
                except (IsADirectoryError, PermissionError, OSError):
                    try:
                        path.rmdir()
                    except OSError:
                        pass
            try:
                self.root.rmdir()
            except OSError:
                pass

    def __enter__(self) -> "ByteSandbox":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # ---------------------------------------------------------------- paths
    def _resolve(self, relative) -> Path:
        if isinstance(relative, (bytes, bytearray)):
            try:
                relative = bytes(relative).decode("utf-8")
            except UnicodeDecodeError as exc:
                raise SandboxViolation("path must be UTF-8 text") from exc
        if not isinstance(relative, str) or not relative:
            raise SandboxViolation("a non-empty relative path is required")
        if relative.startswith(("/", "~")) or (os.path.isabs(relative)):
            raise SandboxViolation("absolute paths are not allowed")
        candidate = (self.root / relative).resolve()
        if candidate != self.root and self.root not in candidate.parents:
            raise SandboxViolation("path escapes the sandbox root")
        if candidate.is_symlink():
            target = candidate.resolve()
            if target != self.root and self.root not in target.parents:
                raise SandboxViolation("symlink escapes the sandbox root")
        return candidate

    # ------------------------------------------------------------ execution
    def _limits(self):
        def apply() -> None:
            cpu = int(self.timeout) + 1
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
            resource.setrlimit(resource.RLIMIT_AS, (MEMORY_LIMIT_BYTES, MEMORY_LIMIT_BYTES))
            resource.setrlimit(resource.RLIMIT_FSIZE, (FILE_SIZE_LIMIT_BYTES, FILE_SIZE_LIMIT_BYTES))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        return apply

    def _environment(self) -> dict[str, str]:
        return {"PATH": os.defpath, "HOME": str(self.root), "TMPDIR": str(self.root),
                "PYTHONIOENCODING": "utf-8", "PYTHONDONTWRITEBYTECODE": "1"}

    def execute_python(self, code_bytes: bytes) -> bytes:
        """Write ``code_bytes`` to a temp file, run it and return stdout+stderr."""
        if isinstance(code_bytes, str):
            code_bytes = code_bytes.encode("utf-8")
        if not isinstance(code_bytes, (bytes, bytearray)):
            raise TypeError("code_bytes must be bytes")
        if not self.enabled:
            return b"ERROR: sandbox execution is disabled."
        if len(code_bytes) > self.max_file_bytes:
            return b"ERROR: submitted code exceeds the sandbox size limit."
        handle = tempfile.NamedTemporaryFile(mode="wb", suffix=".py", dir=str(self.root), delete=False)
        temp_path = handle.name
        try:
            handle.write(bytes(code_bytes))
            handle.flush()
            handle.close()
            try:
                result = subprocess.run(
                    [self.python_executable, "-I", "-B", temp_path],
                    capture_output=True,
                    text=False,  # bytes in, bytes out
                    timeout=self.timeout,
                    cwd=str(self.root),
                    env=self._environment(),
                    preexec_fn=self._limits() if hasattr(resource, "setrlimit") else None,
                )
            except subprocess.TimeoutExpired:
                return b"ERROR: Execution timed out."
            output = result.stdout + STDERR_MARK + result.stderr
            if result.returncode != 0:
                output += b"\n---EXIT---\n" + str(result.returncode).encode("ascii")
            return output[: self.max_output_bytes]
        finally:
            try:
                os.unlink(temp_path)
            except OSError:
                pass

    # ---------------------------------------------------------------- files
    def read_file(self, path_bytes: bytes) -> bytes:
        """Read a file inside the sandbox root; never outside it."""
        try:
            target = self._resolve(path_bytes)
        except SandboxViolation as exc:
            return f"ERROR: {exc}".encode("utf-8")
        if not self.enabled:
            return b"ERROR: sandbox execution is disabled."
        if not target.is_file():
            return f"ERROR: no such file: {target.relative_to(self.root)}".encode("utf-8")
        size = target.stat().st_size
        if size > self.max_file_bytes:
            return b"ERROR: file exceeds the sandbox read limit."
        return target.read_bytes()[: self.max_output_bytes]

    def write_file(self, path_bytes: bytes, content: bytes) -> bytes:
        """Write ``content`` to a path inside the sandbox root atomically."""
        try:
            target = self._resolve(path_bytes)
        except SandboxViolation as exc:
            return f"ERROR: {exc}".encode("utf-8")
        if not self.enabled:
            return b"ERROR: sandbox execution is disabled."
        if len(content) > self.max_file_bytes:
            return b"ERROR: content exceeds the sandbox write limit."
        if target.exists() and target.is_dir():
            return b"ERROR: refusing to overwrite a directory."
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_name(target.name + ".tmp")
        temporary.write_bytes(content)
        os.replace(temporary, target)
        return f"OK: wrote {len(content)} bytes to {target.relative_to(self.root)}".encode("utf-8")

    # ------------------------------------------------------------- dispatch
    def call(self, tool: bytes, argument: bytes) -> bytes:
        """Run an allow-listed tool with a byte argument and return bytes."""
        if tool not in TOOL_NAMES:
            return f"ERROR: unsupported tool {tool!r}".encode("utf-8")
        if not isinstance(argument, (bytes, bytearray)):
            return b"ERROR: tool argument must be bytes"
        argument = bytes(argument)
        if len(argument) > MAX_FILE_BYTES:
            return b"ERROR: tool argument exceeds the sandbox limit."
        if tool == b"exec_python":
            return self.execute_python(argument)
        # write_file frames carry the path on the first line, content after it.
        if tool == b"write_file":
            head, _, content = argument.partition(b"\n")
            return self.write_file(head, content)
        return self.read_file(argument)


def looks_failed(result: bytes) -> bool:
    """Conservative host-side check: did the tool report an error?

    This inspects the sandbox envelope only; it is not code understanding.
    """
    if not result:
        return True
    if result.startswith(b"ERROR:"):
        return True
    marker = result.find(STDERR_MARK)
    if marker < 0:
        return False
    stderr = result[marker + len(STDERR_MARK):]
    stderr = stderr.split(b"\n---EXIT---\n")[0]
    return bool(stderr.strip())
