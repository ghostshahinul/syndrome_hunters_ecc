"""Build a separate C ABI library from clean pinned sources. No package installation."""

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "external"
BUILD = EXT / "fast_cp_build"
PM = "6f63b2b9474ba0fa7e511fe52bffdce858a06984"
STIM = "1320ad7eac7de34d2e9c70daa44fbc6d84174450"


def checkout(folder, url, sha):
    path = EXT / folder
    if not path.exists():
        path.mkdir(parents=True)
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        subprocess.run(
            ["git", "-C", str(path), "fetch", "--depth=1", url, sha], check=True
        )
        subprocess.run(
            ["git", "-C", str(path), "checkout", "--detach", sha], check=True
        )
    rev = subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=no"]
    )
    if rev != sha or dirty:
        raise RuntimeError(f"Expected a clean checkout of {sha} in {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cxx", default="clang++")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument(
        "--detail", action="store_true", help="build the separate stage profiler"
    )
    args = ap.parse_args()
    cxx = shutil.which(args.cxx)
    if not cxx:
        raise RuntimeError(
            "Provide --cxx pointing to clang++ or g++ with a C++20 standard library"
        )
    pm = checkout("cp-upstream", "https://github.com/oscarhiggott/PyMatching.git", PM)
    stim = checkout("cp-stim", "https://github.com/quantumlib/Stim.git", STIM)
    flags = [
        "-std=c++20",
        "-O3",
        "-fno-strict-aliasing",
        "-ffunction-sections",
        "-fdata-sections",
        "-I",
        str(pm / "src"),
        "-I",
        str(stim / "src"),
    ]
    if os.name != "nt":
        flags.extend(["-fPIC", "-fvisibility=hidden"])
    compiler = subprocess.check_output([cxx, "--version"], text=True).splitlines()[0]
    key = hashlib.sha256(repr((compiler, flags, PM, STIM)).encode()).hexdigest()[:12]
    objdir = BUILD / key
    objdir.mkdir(parents=True, exist_ok=True)
    # All dependencies of the upstream DEM reader and Sparse Blossom; omit separate CLIs.
    pm_files = re.search(
        r"set\(SOURCE_FILES_NO_MAIN(.*?)\)",
        (pm / "CMakeLists.txt").read_text(),
        re.DOTALL,
    )[1].split()
    sources = [
        pm / p for p in pm_files if "/diagram/" not in p and "namespaced_main" not in p
    ]
    stim_files = (stim / "file_lists/source_files_no_main").read_text().splitlines()
    sources.extend(
        stim / p
        for p in stim_files
        if any(
            "/" + name + "/" in p
            for name in ("dem", "circuit", "gates", "mem", "util_bot")
        )
    )
    kernel = (
        Path(__file__).with_name("detail.cc" if args.detail else "kernel.cc").resolve()
    )
    sources.append(kernel)

    def compile_one(source):
        name = source.stem + "-" + hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        obj = objdir / (name + ".o")
        if not obj.exists():
            subprocess.run([cxx, *flags, "-c", str(source), "-o", str(obj)], check=True)
        return obj

    with ThreadPoolExecutor(args.jobs) as pool:
        objects = list(pool.map(compile_one, sources))
    windows = os.name == "nt"
    name = "cp_detail" if args.detail else "fast_cp"
    library = BUILD / (name + ".dll" if windows else "lib" + name + ".so")
    link = ["-shared", "-Wl,--gc-sections"] + (
        ["-static"] if windows else ["-Wl,-z,defs"]
    )
    subprocess.run([cxx, *link, *map(str, objects), "-o", str(library)], check=True)
    metadata = {
        "pymatching": PM,
        "stim": STIM,
        "compiler": compiler,
        "platform": platform.platform(),
        "flags": [s for s in flags if str(EXT) not in s and s != "-I"],
        "link_flags": link,
        "kernel_sha256": hashlib.sha256(kernel.read_bytes()).hexdigest(),
        "binary_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
    (BUILD / ("detail_build.json" if args.detail else "build.json")).write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(library)


if __name__ == "__main__":
    main()
