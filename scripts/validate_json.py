#!/usr/bin/env python3
"""Validate JSON files in the repository."""

import json
import glob
import sys


def main():
    errors = []
    for f in glob.glob('models/*.json'):
        try:
            with open(f) as fp:
                json.load(fp)
            print(f"Valid JSON: {f}")
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {f} - {e}")
            errors.append(f)

    if errors:
        print(f"\n{len(errors)} file(s) failed validation")
        sys.exit(1)
    else:
        print("\nAll JSON files valid")


if __name__ == "__main__":
    main()
