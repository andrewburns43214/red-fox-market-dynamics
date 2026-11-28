#!/usr/bin/env python3
"""Set a Team Intelligence provider key from an interactive server terminal."""

import argparse
import getpass
import os
from pathlib import Path
import re
import tempfile
import warnings


PROVIDERS = ("balldontlie", "rad-sports", "big-balls", "puckapi", "ufcalendar")
DIRECTORY = Path("/etc/redfox-team-intelligence/credentials")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=PROVIDERS)
    args = parser.parse_args()
    if os.geteuid() != 0:
        parser.exit(1, "Run this command as root.\n")
    try:
        with open("/dev/tty", "r+", encoding="utf-8") as terminal, warnings.catch_warnings():
            warnings.simplefilter("error", getpass.GetPassWarning)
            key = getpass.getpass(f"Paste the {args.provider} API key: ", stream=terminal).strip()
    except (OSError, EOFError, KeyboardInterrupt, getpass.GetPassWarning):
        parser.exit(1, "An interactive server terminal is required. No key was written.\n")
    if re.fullmatch(r"[A-Za-z0-9._~+/=-]{12,512}", key) is None:
        parser.exit(1, "Key format is invalid. No key was written.\n")

    for directory in (DIRECTORY.parent, DIRECTORY):
        directory.mkdir(mode=0o700, exist_ok=True)
        info = directory.lstat()
        if directory.is_symlink() or info.st_uid != 0 or info.st_mode & 0o077:
            parser.exit(1, "The credential directory must be root-owned and private.\n")

    fd, temporary = tempfile.mkstemp(prefix=".credential-", dir=DIRECTORY)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as output:
            output.write(key + "\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, DIRECTORY / args.provider)
    finally:
        Path(temporary).unlink(missing_ok=True)
    print(f"Saved {args.provider} credential.")


if __name__ == "__main__":
    main()
