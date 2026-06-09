from .diagnostics import MethodAnalysis, analyze_embedding
from .execution import run_benchmark
from .rendering import write_benchmark_report
from .specs import (
	BenchmarkConfig,
	BenchmarkRunResult,
	DatasetRunResult,
	DatasetSpec,
	MethodRunResult,
	MethodSpec,
	load_benchmark_config,
)

generate_benchmark_report = write_benchmark_report

