"""Run the full pipeline: generate data -> clean -> analyse."""
import runpy

for step in ["src/generate_data.py", "src/clean.py", "src/analysis.py"]:
    print(f"\n=== {step} ===")
    runpy.run_path(step, run_name="__main__")
