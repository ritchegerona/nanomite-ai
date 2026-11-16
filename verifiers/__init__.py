"""Deterministic verifiers: the source of training signal for STaR.

Each verifier returns (ok: bool, reason: str). They never call a model.
"""
from .chat_consistency import verify as _chat
from .code_exec import verify as _code
from .security_label import verify as _security
from .tool_call import verify as _tool

_VERIFIERS = {
    "agent": _tool,
    "coding": _code,
    "security": _security,
    "chat": _chat,
}


def verify(domain: str, output: str, meta: dict | None = None) -> tuple[bool, str]:
    fn = _VERIFIERS.get(domain)
    if fn is None:
        return False, f"unknown_domain:{domain}"
    try:
        return fn(output, meta or {})
    except Exception as exc:  # verifier bugs must never crash the pipeline
        return False, f"error:{type(exc).__name__}:{exc}"


__all__ = ["verify"]
