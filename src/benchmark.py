from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Fix sched_getaffinity issue on Windows for scGPT DataLoader compatibility.
if not hasattr(os, "sched_getaffinity"):
	os.sched_getaffinity = lambda pid: set(range(os.cpu_count() or 1))


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
	sys.path.insert(0, str(PROJECT_ROOT))

from src.benchmarking import load_benchmark_config, run_benchmark, write_benchmark_report  # noqa: E402


def main() -> None:
	parser = argparse.ArgumentParser(description="Run modular embedding benchmarks and generate a report")
	parser.add_argument("config", help="YAML benchmark config")
	args = parser.parse_args()

	spec = load_benchmark_config(args.config)
	if spec.project_root is None:
		spec.project_root = str(PROJECT_ROOT)

	run_result = run_benchmark(spec, PROJECT_ROOT)
	report_path = write_benchmark_report(run_result)
	print(f"Benchmark report written to {report_path}")


if __name__ == "__main__":
	main()

