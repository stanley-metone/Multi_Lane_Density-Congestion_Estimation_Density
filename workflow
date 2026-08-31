name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  quality-checks:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12"]

    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          # Core pipeline logic (zones/density/tracker/schemas) has no
          # heavy deps, but pipeline.py imports ultralytics/cv2 lazily --
          # install everything so the full test/lint surface is covered.
          pip install -r requirements-dev.txt
          pip install -e . --no-deps

      - name: Lint with flake8
        run: flake8 src tests scripts --max-line-length=100 --extend-ignore=E203

      - name: Check formatting with black
        run: black --check src tests scripts

      - name: Type-check with mypy
        run: mypy src --ignore-missing-imports
        continue-on-error: true # non-blocking: informative, not a gate yet

      - name: Run tests with coverage
        run: pytest --cov=traffic_density --cov-report=term-missing tests/
