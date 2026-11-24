#!/usr/bin/env python3
"""Run the full analysis pipeline, from the raw data in data/raw/ to every output in outputs/.

    python run_all.py              # steps 1-7
    python run_all.py --download   # also re-download the two API-sourced inputs first (step 0)
    python run_all.py --clean      # delete outputs/ before running

Each step is a separate script in src/ and runs in its own interpreter.
"""
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
STEPS = [
    "01_build_la_dataset.py",
    "02_build_monthly_series.py",
    "03_geographic_analysis.py",
    "04_temporal_analysis.py",
    "05_figures_main.py",
    "06_figures_supplementary.py",
    "07_tables.py",
]

if __name__ == "__main__":
    if "--clean" in sys.argv and (ROOT / "outputs").exists():
        shutil.rmtree(ROOT / "outputs")
    steps = (["00_download_supplementary_sources.py"] if "--download" in sys.argv else []) + STEPS
    t0 = time.time()
    for step in steps:
        t = time.time()
        print(f"=== {step}", flush=True)
        res = subprocess.run([sys.executable, step], cwd=SRC, stdout=subprocess.DEVNULL if "--quiet" in sys.argv else None)
        if res.returncode != 0:
            sys.exit(f"step {step} failed (exit code {res.returncode})")
        print(f"    done in {time.time() - t:.1f} s", flush=True)
    print(f"Pipeline complete in {time.time() - t0:.1f} s. Outputs are in {ROOT / 'outputs'}")
