#!/usr/bin/env python3

import argparse
import io
import subprocess
import tarfile
from datetime import datetime
from pathlib import Path


SOURCE_PATHS = (
    "CMakeLists.txt", "LICENSE", "AUTHORS", "README.md",
    "app", "cmake", "common", "ggml", "include", "licenses", "src", "vendor", "tools",
    "grammars", "models/templates",
    "scripts/package_sources.py", "scripts/build-info.sh",
    "scripts/ui-assets.cmake", "scripts/connect-download.cmake",
)
EXCLUDED_DIRS = {"docs", "examples", "tests", "test", "bench", "benchmarks", "pocs", "media", "node_modules", "__pycache__", ".github"}
UI_BUILD_FILES = {"tools/ui/CMakeLists.txt", "tools/ui/ui.cpp.in", "tools/ui/ui.h.in"}


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def main():
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Package native build sources without Web UI, examples, tests or models.")
    parser.add_argument("output_dir", nargs="?", type=Path, default=root, help="archive directory (default: repository root)")
    args = parser.parse_args()

    if Path(git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve() != root:
        parser.error("run this script from a Git checkout of llama.cpp")

    paths = git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z", "--", *SOURCE_PATHS)
    files = []
    for raw_path in sorted(set(paths.split(b"\0")) - {b""}):
        name = raw_path.decode("utf-8")
        path = Path(name)
        if EXCLUDED_DIRS.intersection(path.parts[:-1]):
            continue
        if path.match("llama.cpp-*.tar.gz"):
            continue
        if name.startswith("tools/ui/") and name not in UI_BUILD_FILES:
            continue
        if (root / path).is_file() or (root / path).is_symlink():
            files.append(name)

    commit = git(root, "rev-parse", "--short", "HEAD").decode().strip()
    if git(root, "status", "--porcelain", "--", *SOURCE_PATHS).strip():
        commit += "-dirty"
    build_number = git(root, "rev-list", "--count", "HEAD").decode().strip()
    cache = (
        '# Initial cache for a native build without Web UI, examples or tests.\n'
        'set(LLAMA_BUILD_UI OFF CACHE BOOL "")\n'
        'set(LLAMA_USE_PREBUILT_UI OFF CACHE BOOL "")\n'
        'set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "")\n'
        'set(LLAMA_BUILD_TESTS OFF CACHE BOOL "")\n'
        f'set(LLAMA_BUILD_NUMBER {build_number} CACHE STRING "")\n'
        f'set(LLAMA_BUILD_COMMIT "{commit}" CACHE STRING "")\n'
    ).encode()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / f"llama.cpp-{datetime.now():%Y%m%d%H%M}.tar.gz"
    try:
        output = archive_path.open("xb")
    except FileExistsError:
        parser.error(f"archive already exists: {archive_path}")

    try:
        with output, tarfile.open(fileobj=output, mode="w:gz") as archive:
            for name in files:
                archive.add(root / name, arcname=f"llama.cpp/{name}", recursive=False)
            info = tarfile.TarInfo("llama.cpp/source-build.cmake")
            info.size = len(cache)
            info.mode = 0o644
            archive.addfile(info, io.BytesIO(cache))
    except BaseException:
        archive_path.unlink()
        raise

    print(archive_path)
    print(f"{len(files)} source files; revision {commit}")
    print("After extraction: cmake -S llama.cpp -B build -C llama.cpp/source-build.cmake [backend options]")


if __name__ == "__main__":
    main()
