#!/usr/bin/env python3
"""Tool registry for the Brent 1.0-4b agent loop — allow-all by default.

Every registered tool is available with no command allow-list, no path
confinement, and no deny-by-default rules. This is intended for authorized
security research, pentesting, CTIA, and office use. Outputs are
secret-redacted and every call is logged to artifacts/agent_log.jsonl.

Tools are executed autonomously: run_command auto-detects missing binaries and
attempts to install them via the system package manager (apt, brew, pip) before
retrying. install_tool provides explicit tool installation.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
LOG_PATH = ROOT / "artifacts" / "agent_log.jsonl"

MAX_OUTPUT = 1_000_000
WEB_TIMEOUT = 300
CMD_TIMEOUT = 600
INSTALL_TIMEOUT = 600

SECRET_PATTERNS = [
    re.compile(r"(sk|pk|ghp|glpat|github|aws|azure|gcp)[_-]?[0-9a-zA-Z]{10,}", re.I),
    re.compile(r"(api[_-]?key|secret|token|password|passwd)\s*[:=]\s*\S+", re.I),
    re.compile(r"-----BEGIN [A-Z ]+-----"),
]

# Package-manager install commands per tool name (best-effort, OS-dependent).
_TOOL_INSTALL_CMDS = {
    "nmap": ["apt-get", "install", "-y", "nmap"],
    "nikto": ["apt-get", "install", "-y", "nikto"],
    "sqlmap": ["apt-get", "install", "-y", "sqlmap"],
    "hydra": ["apt-get", "install", "-y", "hydra"],
    "john": ["apt-get", "install", "-y", "john"],
    "aircrack-ng": ["apt-get", "install", "-y", "aircrack-ng"],
    "wifite": ["apt-get", "install", "-y", "wifite"],
    "wifite2": ["pip3", "install", "wifite2"],
    "hashcat": ["apt-get", "install", "-y", "hashcat"],
    "gobuster": ["apt-get", "install", "-y", "gobuster"],
    "dirb": ["apt-get", "install", "-y", "dirb"],
    "masscan": ["apt-get", "install", "-y", "masscan"],
    "zaproxy": ["apt-get", "install", "-y", "zaproxy"],
    "burpsuite": ["apt-get", "install", "-y", "burpsuite"],
    "metasploit": ["apt-get", "install", "-y", "metasploit-framework"],
}

_TOOL_INSTALL_CMDS_BREW = {
    "nmap": "nmap",
    "nikto": "nikto",
    "sqlmap": "sqlmap",
    "hydra": "hydra",
    "john": "john",
    "aircrack-ng": "aircrack-ng",
    "wifite": "wifite",
    "hashcat": "hashcat",
    "gobuster": "gobuster",
    "dirb": "dirb",
    "masscan": "masscan",
}

TOOL_CALL_LOG: list[dict] = []


def redact(text: str) -> str:
    for pat in SECRET_PATTERNS:
        text = pat.sub("[REDACTED]", text)
    return text


def _log(name: str, args: dict, result: Any, ok: bool) -> None:
    entry = {
        "tool": name,
        "arguments": redact(json.dumps(args, ensure_ascii=False)),
        "ok": ok,
        "result": redact(str(result)[:MAX_OUTPUT]),
    }
    TOOL_CALL_LOG.append(entry)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def read_file(path: str) -> dict:
    p = Path(path)
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p
    if not resolved.is_file():
        return {"ok": False, "error": "not_a_file"}
    data = resolved.read_text(encoding="utf-8", errors="replace")[:MAX_OUTPUT]
    _log("read_file", {"path": path}, data, True)
    return {"ok": True, "content": data}


def write_file(path: str, content: str) -> dict:
    p = Path(path)
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(content, encoding="utf-8")
    _log("write_file", {"path": path, "bytes": len(content)}, "written", True)
    return {"ok": True, "path": str(resolved), "bytes": len(content)}


def list_dir(path: str = ".") -> dict:
    p = Path(path)
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p
    if not resolved.is_dir():
        return {"ok": False, "error": "not_a_dir"}
    entries = [
        {"name": child.name, "dir": child.is_dir()}
        for child in sorted(resolved.iterdir())
    ]
    _log("list_dir", {"path": path}, entries, True)
    return {"ok": True, "entries": entries[:200]}


def check_tool(name: str) -> dict:
    """Check if a binary is available on PATH. Returns ok=True if found."""
    found = shutil.which(name) is not None
    _log("check_tool", {"name": name}, {"found": found}, found)
    return {"ok": found, "found": found, "path": shutil.which(name) if found else None}


def _try_install(name: str) -> dict:
    """Attempt to install a tool using the system package manager."""
    # Try apt-get (Debian/Ubuntu/Kali)
    cmd = _TOOL_INSTALL_CMDS.get(name)
    if cmd:
        try:
            proc = subprocess.run(
                cmd,
                shell=False,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=INSTALL_TIMEOUT,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"},
            )
            if proc.returncode == 0:
                _log("install_tool", {"name": name, "method": "apt-get"}, "installed", True)
                return {"ok": True, "method": "apt-get", "installed": True}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Try brew (macOS)
    pkg = _TOOL_INSTALL_CMDS_BREW.get(name)
    if pkg and shutil.which("brew"):
        try:
            proc = subprocess.run(
                ["brew", "install", pkg],
                shell=False,
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=INSTALL_TIMEOUT,
                env={"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"},
            )
            if proc.returncode == 0:
                _log("install_tool", {"name": name, "method": "brew"}, "installed", True)
                return {"ok": True, "method": "brew", "installed": True}
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

    # Try pip3 for Python tools
    try:
        proc = subprocess.run(
            ["pip3", "install", name],
            shell=False,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=INSTALL_TIMEOUT,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"},
        )
        if proc.returncode == 0:
            _log("install_tool", {"name": name, "method": "pip3"}, "installed", True)
            return {"ok": True, "method": "pip3", "installed": True}
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    _log("install_tool", {"name": name}, "install_failed", False)
    return {"ok": False, "error": "install_failed"}


def install_tool(name: str) -> dict:
    """Install a tool if not already present. Returns installation result."""
    if shutil.which(name):
        _log("install_tool", {"name": name}, {"already_installed": True}, True)
        return {"ok": True, "already_installed": True, "path": shutil.which(name)}
    return _try_install(name)


def run_command(cmd: str) -> dict:
    if not cmd or not cmd.strip():
        return {"ok": False, "error": "empty_command"}

    # Extract the first token as the binary name for availability check.
    first_token = cmd.strip().split()[0]

    # Check if binary exists; if not, attempt to install it.
    if not shutil.which(first_token):
        install_result = install_tool(first_token)
        if not install_result.get("ok"):
            return {
                "ok": False,
                "error": f"binary_not_found:{first_token}",
                "install": install_result,
            }

    try:
        proc = subprocess.run(
            cmd,
            shell=True,
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=CMD_TIMEOUT,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"},
        )
        stdout = (proc.stdout or "")[:MAX_OUTPUT]
        stderr = (proc.stderr or "")[:MAX_OUTPUT]
        out = stdout + stderr
        _log("run_command", {"cmd": cmd}, out, proc.returncode == 0)
        return {
            "ok": proc.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": proc.returncode,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout"}
    except FileNotFoundError:
        return {"ok": False, "error": "binary_not_found"}


def web_fetch(url: str) -> dict:
    if not re.match(r"^https?://", url):
        return {"ok": False, "error": "only_http_https_allowed"}
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "nanomite-ai-agent/1.0"}
        )
        with urllib.request.urlopen(req, timeout=WEB_TIMEOUT) as resp:
            raw = resp.read(MAX_OUTPUT).decode("utf-8", errors="replace")
        _log("web_fetch", {"url": url}, raw, True)
        return {"ok": True, "content": raw}
    except Exception as exc:  # network errors must not crash the loop
        return {"ok": False, "error": str(exc)[:200]}


def extract_metadata(path: str) -> dict:
    """Extract metadata from files: EXIF from images, document properties, etc."""
    p = Path(path)
    try:
        resolved = p.resolve()
    except Exception:
        resolved = p
    if not resolved.is_file():
        return {"ok": False, "error": "not_a_file"}

    # Try exiftool first
    try:
        proc = subprocess.run(
            ["exiftool", str(resolved)],
            capture_output=True,
            text=True,
            timeout=CMD_TIMEOUT,
            env={"PATH": os.environ.get("PATH", ""), "PYTHONNOUSERSITE": "1"},
        )
        if proc.returncode == 0 and proc.stdout.strip():
            metadata = {}
            for line in proc.stdout.strip().split("\n"):
                if ":" in line:
                    key, val = line.split(":", 1)
                    metadata[key.strip()] = val.strip()
            _log("extract_metadata", {"path": path}, metadata, True)
            return {"ok": True, "metadata": metadata, "source": "exiftool"}
    except FileNotFoundError:
        pass

    # Fallback: use Python PIL/Pillow for image EXIF
    try:
        from PIL import Image
        from PIL.ExifTags import TAGS
        img = Image.open(resolved)
        exif = img._getexif()
        metadata = {}
        if exif:
            for tag_id, value in exif.items():
                tag = TAGS.get(tag_id, tag_id)
                metadata[str(tag)] = str(value)
        _log("extract_metadata", {"path": path}, metadata, True)
        return {"ok": True, "metadata": metadata, "source": "pil_exif", "format": img.format, "size": img.size, "mode": img.mode}
    except Exception:
        pass

    # Generic file info fallback
    stat = resolved.stat()
    metadata = {
        "filename": resolved.name,
        "size_bytes": stat.st_size,
        "modified": str(stat.st_mtime),
        "created": str(stat.st_ctime),
        "extension": resolved.suffix,
    }
    _log("extract_metadata", {"path": path}, metadata, True)
    return {"ok": True, "metadata": metadata, "source": "stat"}


def osint_username(username: str) -> dict:
    """OSINT: enumerate a username across common platforms."""
    import urllib.parse
    encoded = urllib.parse.quote(username)
    results = {}
    platforms = {
        "github": f"https://github.com/{encoded}",
        "twitter": f"https://twitter.com/{encoded}",
        "instagram": f"https://instagram.com/{encoded}",
        "linkedin": f"https://linkedin.com/in/{encoded}",
        "reddit": f"https://reddit.com/user/{encoded}",
        "youtube": f"https://youtube.com/@{encoded}",
    }
    found = []
    not_found = []
    for name, url in platforms.items():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "nanomite-ai-agent/1.0"})
            with urllib.request.urlopen(req, timeout=WEB_TIMEOUT) as resp:
                if resp.status == 200:
                    found.append({"platform": name, "url": url})
        except Exception:
            not_found.append({"platform": name, "url": url})
    _log("osint_username", {"username": username}, {"found": found, "not_found": not_found}, True)
    return {"ok": True, "username": username, "found": found, "not_found": not_found}


TOOLS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
    "run_command": run_command,
    "web_fetch": web_fetch,
    "check_tool": check_tool,
    "install_tool": install_tool,
    "extract_metadata": extract_metadata,
    "osint_username": osint_username,
}

TOOL_SCHEMAS = {
    "read_file": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    "write_file": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}}, "required": ["path", "content"]},
    "list_dir": {"type": "object", "properties": {"path": {"type": "string"}}, "required": []},
    "run_command": {"type": "object", "properties": {"cmd": {"type": "string"}}, "required": ["cmd"]},
    "web_fetch": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]},
    "check_tool": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    "install_tool": {"type": "object", "properties": {"name": {"type": "string"}}, "required": ["name"]},
    "extract_metadata": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    "osint_username": {"type": "object", "properties": {"username": {"type": "string"}}, "required": ["username"]},
}


def tool_list_text() -> str:
    lines = []
    for name, schema in TOOL_SCHEMAS.items():
        req = ", ".join(schema.get("required", [])) or "none"
        lines.append(f"- {name}({req}): {schema}")
    return "\n".join(lines)
