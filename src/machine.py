"""Machine fingerprint for results CSVs: CPU model, frequency, power state, threads, git commit.

Timings on a laptop drift with clock speed and power plan, so every benchmark row records
the state of the machine when it was measured. All probes are best-effort and return
"unknown" rather than raising.
"""

from __future__ import annotations

import ctypes
import os
import platform
import subprocess
import sys
from functools import lru_cache


def _run(cmd: list[str]) -> str:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


@lru_cache(maxsize=1)
def cpu_model() -> str:
    if sys.platform == "win32":
        try:
            import winreg

            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                 r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
            return str(winreg.QueryValueEx(key, "ProcessorNameString")[0]).strip()
        except OSError:
            pass
    if os.path.exists("/proc/cpuinfo"):
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    return platform.processor() or "unknown"


def cpu_mhz() -> str:
    """Current clock (MHz) as Windows reports it; 'unknown' elsewhere."""
    if sys.platform == "win32":
        out = _run(["powershell", "-NoProfile", "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1).CurrentClockSpeed"])
        if out.isdigit():
            return out
    return "unknown"


def power_state() -> str:
    """'AC' or 'battery NN%' plus the active power plan name (Windows); 'unknown' elsewhere."""
    if sys.platform != "win32":
        return "unknown"

    class _Status(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]

    st = _Status()
    line = "unknown"
    if ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st)):
        line = "AC" if st.ACLineStatus == 1 else f"battery {st.BatteryLifePercent}%"
    plan = _run(["powercfg", "/getactivescheme"])
    if "(" in plan:
        line += " / " + plan[plan.rfind("(") + 1:plan.rfind(")")]
    overlay = _power_overlay()
    if overlay:
        line += f" / mode {overlay}"
    return line


OVERLAYS = {"ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
            "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
            "00000000-0000-0000-0000-000000000000": "Balanced"}


def _power_overlay() -> str:
    """Windows 11 power mode (overlay on the plan) for the current power source."""
    try:
        import winreg

        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                             r"SYSTEM\CurrentControlSet\Control\Power\User\PowerSchemes")
        st = _Status_ac()
        name = "ActiveOverlayAcPowerScheme" if st else "ActiveOverlayDcPowerScheme"
        guid = str(winreg.QueryValueEx(key, name)[0]).lower()
        return OVERLAYS.get(guid, guid)
    except OSError:
        return ""


def _Status_ac() -> bool:
    class _S(ctypes.Structure):
        _fields_ = [("ACLineStatus", ctypes.c_byte), ("BatteryFlag", ctypes.c_byte),
                    ("BatteryLifePercent", ctypes.c_byte), ("SystemStatusFlag", ctypes.c_byte),
                    ("BatteryLifeTime", ctypes.c_ulong), ("BatteryFullLifeTime", ctypes.c_ulong)]
    st = _S()
    return bool(ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(st))) and st.ACLineStatus == 1


def clear_numba_cache(root: str | os.PathLike | None = None) -> int:
    """Delete numba's on-disk cache (*.nbi, *.nbc) under src/. Its index does not notice edits
    to callees in other files, so a stale cache can crash or mis-time a kernel. Returns the
    number of files removed. Call before importing/compiling kernels for a timing run."""
    from pathlib import Path

    base = Path(root) if root else Path(__file__).resolve().parent
    n = 0
    for pattern in ("*.nbi", "*.nbc"):
        for f in base.rglob(pattern):
            try:
                f.unlink()
                n += 1
            except OSError:
                pass
    return n


@lru_cache(maxsize=1)
def logical_cpus() -> int:
    return os.cpu_count() or 1


def git_commit() -> str:
    out = _run(["git", "rev-parse", "--short", "HEAD"])
    if not out:
        return "unknown"
    dirty = _run(["git", "status", "--porcelain", "--untracked-files=no"])
    return out + ("+dirty" if dirty else "")


def fingerprint() -> dict[str, str]:
    """Snapshot of the machine state; call once per benchmark run."""
    return {"cpu": cpu_model(), "cpu_mhz": cpu_mhz(), "power": power_state(),
            "logical_cpus": str(logical_cpus()), "git_commit": git_commit()}


def short_cpu(name: str | None = None) -> str:
    """Compact CPU name for figure footers, e.g. 'i5-13420H'."""
    name = name or cpu_model()
    for tok in name.replace("(R)", "").replace("(TM)", "").split():
        if any(c.isdigit() for c in tok) and "-" in tok:
            return tok
    return name
