"""Entry point for running from source (python run.py) and for PyInstaller."""

from vinted_prep.app import main

if __name__ == "__main__":
    raise SystemExit(main())
