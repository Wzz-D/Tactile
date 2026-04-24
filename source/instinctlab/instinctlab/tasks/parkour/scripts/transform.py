#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickletools
import shutil
import struct
from pathlib import Path


DEFAULT_REPLACEMENTS = [
    ("/home/aigc/Wzz/instinctlab", "/home/future/instinct/instinctlab"),
    ("/home/aigc/Wzz/Datasets", "/home/future/instinct/Datasets"),
]

_STRING_OPS = {
    "SHORT_BINUNICODE",
    "BINUNICODE",
    "BINUNICODE8",
    "UNICODE",
}


def _rewrite_string(value: str, replacements: list[tuple[str, str]]) -> str:
    result = value
    for old, new in replacements:
        if old in result:
            result = result.replace(old, new)
    return result


def _encode_unicode_opcode(value: str) -> bytes:
    encoded = value.encode("utf-8")
    size = len(encoded)
    if size < 256:
        return b"\x8c" + bytes((size,)) + encoded
    if size <= 0xFFFFFFFF:
        return b"X" + struct.pack("<I", size) + encoded
    return b"\x8d" + struct.pack("<Q", size) + encoded


def _scan_and_patch_pickle(raw: bytes, replacements: list[tuple[str, str]]) -> tuple[bytes, list[tuple[str, str]], list[str]]:
    ops = list(pickletools.genops(raw))
    out = bytearray()
    changed_strings: list[tuple[str, str]] = []
    remaining_old_strings: set[str] = set()

    for index, (opcode, arg, pos) in enumerate(ops):
        next_pos = ops[index + 1][2] if index + 1 < len(ops) else len(raw)
        op_name = opcode.name

        if op_name == "FRAME":
            continue

        if op_name in _STRING_OPS and isinstance(arg, str):
            new_arg = _rewrite_string(arg, replacements)
            if new_arg != arg:
                changed_strings.append((arg, new_arg))
            if "/home/aigc/Wzz/" in new_arg:
                remaining_old_strings.add(new_arg)
            out.extend(_encode_unicode_opcode(new_arg))
            continue

        out.extend(raw[pos:next_pos])

    return bytes(out), changed_strings, sorted(remaining_old_strings)


def _patch_env_pkl(env_pkl_path: Path, replacements: list[tuple[str, str]]) -> None:
    if not env_pkl_path.is_file():
        raise FileNotFoundError(f"env.pkl not found: {env_pkl_path}")

    raw = env_pkl_path.read_bytes()
    patched_raw, changed_strings, remaining_old = _scan_and_patch_pickle(raw, replacements)

    backup_path = env_pkl_path.with_suffix(env_pkl_path.suffix + ".bak")
    if backup_path.exists():
        raise FileExistsError(f"Backup already exists, refusing to overwrite: {backup_path}")
    shutil.copy2(env_pkl_path, backup_path)
    env_pkl_path.write_bytes(patched_raw)

    print(f"[OK] patched: {env_pkl_path}")
    print(f"[OK] backup : {backup_path}")
    if changed_strings:
        for old_path, new_path in changed_strings:
            print(f"  OLD: {old_path}")
            print(f"  NEW: {new_path}")
    else:
        print("  [INFO] no matching path strings found")

    if remaining_old:
        print("[WARN] remaining old paths:")
        for path in remaining_old:
            print(f"  {path}")
    else:
        print("[OK] no remaining /home/aigc/Wzz paths")


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch saved env.pkl absolute paths after moving runs between machines.")
    parser.add_argument("env_pkl", nargs="+", help="One or more params/env.pkl files to patch in place.")
    args = parser.parse_args()

    for env_pkl in args.env_pkl:
        _patch_env_pkl(Path(env_pkl).expanduser().resolve(), DEFAULT_REPLACEMENTS)


if __name__ == "__main__":
    main()
