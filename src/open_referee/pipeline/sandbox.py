"""Sandboxed sympy evaluation for formula recomputation.

Runs expressions in an isolated ``python -I`` subprocess: empty environment
(no network credentials, no PATH), private temp cwd, CPU/memory/file-size
rlimits, hard timeout, and a strict input filter before anything is sent.
The runner imports sympy only, parses each expression with a whitelisted
locals table, and prints a JSON array of results.
"""

from __future__ import annotations

import json
import logging
import re
import resource
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)

RUNNER = r"""
import json, sys
data = json.load(sys.stdin)
try:
    import sympy
except Exception:
    print(json.dumps([{"ok": False, "error": "sympy unavailable"}] * len(data)))
    sys.exit(0)

SAFE = {
    "sqrt": sympy.sqrt, "log": sympy.log, "ln": sympy.log, "exp": sympy.exp,
    "abs": sympy.Abs, "Min": sympy.Min, "Max": sympy.Max, "sum": sympy.Sum,
    "binomial": sympy.binomial, "factorial": sympy.factorial,
    "sin": sympy.sin, "cos": sympy.cos, "tan": sympy.tan,
    "pi": sympy.pi, "E": sympy.E,
}
out = []
for item in data:
    expr = item.get("expression", "")
    subs = item.get("subs") or {}
    try:
        symbols = {k: sympy.Symbol(k) for k in subs}
        e = sympy.sympify(expr, locals={**SAFE, **symbols}, evaluate=True)
        if subs:
            e = e.subs({symbols[k]: sympy.Float(str(v), 15) for k, v in subs.items()})
        v = sympy.N(e, 12)
        out.append({"ok": True, "value": float(v)})
    except Exception as ex:
        out.append({"ok": False, "error": str(ex)[:180]})
sys.stdout.write(json.dumps(out)[:65536])
"""

_BANNED_RE = re.compile(
    r"__|import|lambda|eval|exec|open\s*\(|input\s*\(|getattr|setattr|"
    r"globals|locals\s*\(|compile|os\.|sys\.|subprocess|open\{|\\\n"
)
MAX_EXPR_LEN = 500
MAX_ITEMS = 50
DEFAULT_TIMEOUT_S = 10
MEM_LIMIT_BYTES = 512 * 1024 * 1024


class SandboxDisabled(RuntimeError):
    """Raised when the operator has disabled code execution."""


def _validate(item: dict) -> None:
    expr = item.get("expression", "")
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("empty expression")
    if len(expr) > MAX_EXPR_LEN:
        raise ValueError("expression too long")
    if _BANNED_RE.search(expr):
        raise ValueError("expression contains banned tokens")
    subs = item.get("subs") or {}
    if not isinstance(subs, dict) or len(subs) > 40:
        raise ValueError("bad substitution table")
    for k, v in subs.items():
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,15}", str(k)):
            raise ValueError(f"bad symbol name: {k!r}")
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            raise ValueError(f"bad value for {k}: {v!r}")


def _limits() -> None:  # preexec_fn in the child process
    resource.setrlimit(resource.RLIMIT_CPU, (DEFAULT_TIMEOUT_S, DEFAULT_TIMEOUT_S))
    resource.setrlimit(resource.RLIMIT_AS, (MEM_LIMIT_BYTES, MEM_LIMIT_BYTES))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1024 * 1024, 1024 * 1024))


def evaluate(items: list[dict], *, timeout_s: int = DEFAULT_TIMEOUT_S) -> list[dict]:
    """Evaluate ``[{"expression": str, "subs": {name: number}}]`` in the sandbox.

    Returns one ``{"ok": bool, "value"?: float, "error"?: str}`` per item;
    validation failures and subprocess crashes degrade to per-item errors,
    never exceptions."""
    if not items:
        return []
    items = items[:MAX_ITEMS]
    results: list[dict] = []
    for item in items:
        try:
            _validate(item)
            results.append(None)  # placeholder
        except ValueError as e:
            results.append({"ok": False, "error": f"rejected: {e}"})
    pending = [it for it, r in zip(items, results, strict=True) if r is None]
    if pending:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-c", RUNNER],
                input=json.dumps(pending).encode(),
                capture_output=True,
                timeout=timeout_s + 5,
                env={},
                cwd=tempfile.mkdtemp(prefix="or-sbx-"),
                preexec_fn=_limits,
            )
            batch = json.loads(proc.stdout.decode(errors="replace") or "[]")
        except subprocess.TimeoutExpired:
            batch = [{"ok": False, "error": "sandbox timeout"}] * len(pending)
        except Exception as e:  # noqa: BLE001 - degrade, never crash the run
            logger.warning("sandbox evaluation failed: %s", e)
            batch = [{"ok": False, "error": f"sandbox error: {e}"}] * len(pending)
        it = iter(batch)
        results = [
            r if r is not None else (next(it) or {"ok": False, "error": "no result"})
            for r in results
        ]
    return results
