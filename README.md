# local-llm-benchmark

Lightweight benchmark script for comparing local LLM responsiveness on your own hardware (Ollama and/or MLX).

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Ollama](https://img.shields.io/badge/Backend-Ollama-000000?style=flat-square)
![MLX](https://img.shields.io/badge/Backend-MLX-0A84FF?style=flat-square)
![Mode](https://img.shields.io/badge/
![Output](https://img.shields.io/badge/Output-JSONL-F39C12?style=flat-square)

## Why This Exists

- Compare real-world responsiveness across local models on the same machine.
- Keep benchmarking simple: select only the models you want each run.
- Prioritize warm performance by default (cold is opt-in).

## What It Measures

- `avg_total_s`: average total response latency.
- `p50_total_s`: median total response latency.
- `avg_ttft_s`: average time-to-first-token (available for Ollama path).
- `avg_prompt_tps`: prompt token throughput when backend reports prompt timing.
- `avg_gen_tps`: generation token throughput.
- `avg_load_s`: average model load time (primarily relevant for cold runs).

The script prints separate tables for warm and cold summaries.

## Requirements

- Python 3.10+
- `requests`
- Running local backend(s):
	- Ollama at `http://127.0.0.1:11434`
	- MLX OpenAI-compatible server at `http://127.0.0.1:11435`

Install dependency:

```bash
python3 -m pip install requests
```

## Quick Start

Warm-only benchmark (default):

```bash
python3 benchmark.py
```

Warm + cold benchmark:

```bash
python3 benchmark.py --include-cold
```

List discovered models and exit:

```bash
python3 benchmark.py --backend both --list-models
```

Select backend explicitly:

```bash
python3 benchmark.py --backend ollama
python3 benchmark.py --backend mlx
python3 benchmark.py --backend both
```

Pin exact models (repeat flags):

```bash
python3 benchmark.py \
	--backend both \
	--ollama-model qwen3.5:9b-fast \
	--ollama-model gemma4:26b \
	--mlx-model mlx-community/Meta-Llama-3.1-8B-Instruct-4bit
```

Tune repetitions and output path:

```bash
python3 benchmark.py --warm-runs 5 --output bench_results.jsonl
```

## Interactive Selection Behavior

- If you provide `--ollama-model` / `--mlx-model`, only those are benchmarked.
- If you omit model flags, the script discovers available models and prompts you to choose by number.
- In non-interactive environments, model flags are required.

## Fairness Notes

- Use warm results for responsiveness comparisons.
- Cold results are useful for startup behavior only.
- Keep selected models and prompt set consistent across runs.
- Run multiple warm repetitions to reduce noise.

## Output Files

- `bench_results.jsonl`: one JSON row per request run.
- Console summary: grouped by backend+model, with warm table first.

## Safety Note

Default URLs use loopback (`127.0.0.1`), which is local-only and not externally routable.
