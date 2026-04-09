#!/usr/bin/env python3

import argparse
import json
import statistics
import sys
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

import requests


PROMPTS = [
    "Explain in 5 bullet points how a vector database differs from a relational database.",
    "Write a Python function that returns the prime factors of an integer.",
    "Summarize the advantages and disadvantages of unified memory on Apple Silicon.",
    "Give a concise explanation of PID control for an engineer.",
]

MAX_TOKENS = 256
TEMPERATURE = 0.0
REPEAT_WARM_RUNS = 3
TIMEOUT = 300

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_MLX_URL = "http://127.0.0.1:11435"


@dataclass
class RunResult:
    backend: str
    model: str
    prompt_index: int
    cold: bool
    prompt_chars: int
    output_chars: int
    prompt_tokens: Optional[int]
    output_tokens: Optional[int]
    ttft_s: Optional[float]
    total_s: float
    prompt_tps: Optional[float]
    gen_tps: Optional[float]
    load_s: Optional[float]
    raw_meta: Dict[str, Any]


def mean_or_none(values: List[Optional[float]]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark local models on Ollama, MLX, or both."
    )
    parser.add_argument(
        "--backend",
        choices=["ollama", "mlx", "both"],
        default="ollama",
        help="Which backend(s) to benchmark.",
    )
    parser.add_argument(
        "--ollama-url",
        default=DEFAULT_OLLAMA_URL,
        help="Base URL for Ollama.",
    )
    parser.add_argument(
        "--mlx-url",
        default=DEFAULT_MLX_URL,
        help="Base URL for MLX OpenAI-compatible server.",
    )
    parser.add_argument(
        "--ollama-model",
        action="append",
        default=[],
        help="Ollama model name. Repeat the flag for multiple models.",
    )
    parser.add_argument(
        "--mlx-model",
        action="append",
        default=[],
        help="MLX model name. Repeat the flag for multiple models.",
    )
    parser.add_argument(
        "--warm-runs",
        type=int,
        default=REPEAT_WARM_RUNS,
        help="Number of warm repetitions for each prompt/model.",
    )
    parser.add_argument(
        "--include-cold",
        action="store_true",
        help="Include cold runs before warm runs (default is warm-only).",
    )
    parser.add_argument(
        "--output",
        default="bench_results.jsonl",
        help="Path for JSONL benchmark output.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List discovered models for selected backend(s) and exit.",
    )
    return parser.parse_args()


def discover_ollama_models(base_url: str) -> List[str]:
    url = f"{base_url}/api/tags"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    models = data.get("models", [])
    names = [m.get("name") for m in models if m.get("name")]
    return names


def discover_openai_compat_models(base_url: str) -> List[str]:
    url = f"{base_url}/v1/models"
    response = requests.get(url, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    models = data.get("data", [])
    ids = [m.get("id") for m in models if m.get("id")]
    return ids


def _resolve_models(explicit: List[str], discovered: List[str]) -> List[str]:
    if explicit:
        return explicit
    return discovered


def select_models_interactively(backend_name: str, discovered: List[str]) -> List[str]:
    if not discovered:
        return []

    print(f"\nSelect {backend_name} model(s) to benchmark:")
    for idx, model in enumerate(discovered, start=1):
        print(f"  {idx}. {model}")

    while True:
        raw = input("Enter comma-separated numbers (or 'all'): ").strip().lower()
        if raw == "all":
            return discovered
        if not raw:
            print("Please choose at least one model.")
            continue

        tokens = [t.strip() for t in raw.split(",") if t.strip()]
        if not tokens:
            print("Invalid selection. Try again.")
            continue

        selected_indexes: List[int] = []
        valid = True
        for token in tokens:
            if not token.isdigit():
                valid = False
                break
            idx = int(token)
            if idx < 1 or idx > len(discovered):
                valid = False
                break
            selected_indexes.append(idx - 1)

        if not valid:
            print("Invalid selection. Use numbers from the list, for example: 1,3")
            continue

        # Preserve order and remove duplicates.
        seen = set()
        selected_models: List[str] = []
        for idx in selected_indexes:
            if idx not in seen:
                seen.add(idx)
                selected_models.append(discovered[idx])
        return selected_models


def print_discovered_models(args: argparse.Namespace) -> None:
    if args.backend in ("ollama", "both"):
        try:
            models = discover_ollama_models(args.ollama_url)
            print("Ollama models:")
            if not models:
                print("  (none found)")
            for model in models:
                print(f"  - {model}")
        except requests.RequestException as exc:
            print(f"Ollama models: failed to query {args.ollama_url} ({exc})")

    if args.backend in ("mlx", "both"):
        try:
            models = discover_openai_compat_models(args.mlx_url)
            print("MLX models:")
            if not models:
                print("  (none found)")
            for model in models:
                print(f"  - {model}")
        except requests.RequestException as exc:
            print(f"MLX models: failed to query {args.mlx_url} ({exc})")


def build_backends(args: argparse.Namespace) -> List[Dict[str, str]]:
    backends: List[Dict[str, str]] = []

    discovered_ollama: List[str] = []
    discovered_mlx: List[str] = []

    if args.backend in ("ollama", "both") and not args.ollama_model:
        try:
            discovered_ollama = discover_ollama_models(args.ollama_url)
        except requests.RequestException as exc:
            print(f"Warning: could not auto-discover Ollama models at {args.ollama_url}: {exc}")

    if args.backend in ("mlx", "both") and not args.mlx_model:
        try:
            discovered_mlx = discover_openai_compat_models(args.mlx_url)
        except requests.RequestException as exc:
            print(f"Warning: could not auto-discover MLX models at {args.mlx_url}: {exc}")

    if discovered_ollama and not args.ollama_model:
        if sys.stdin.isatty():
            discovered_ollama = select_models_interactively("Ollama", discovered_ollama)
        else:
            raise ValueError("No --ollama-model provided in non-interactive mode.")

    if discovered_mlx and not args.mlx_model:
        if sys.stdin.isatty():
            discovered_mlx = select_models_interactively("MLX", discovered_mlx)
        else:
            raise ValueError("No --mlx-model provided in non-interactive mode.")

    ollama_models = _resolve_models(args.ollama_model, discovered_ollama)
    mlx_models = _resolve_models(args.mlx_model, discovered_mlx)

    if args.backend in ("ollama", "both"):
        if not ollama_models:
            raise ValueError(
                "No Ollama models found. Use --ollama-model or run with --list-models to verify discovery."
            )
        for model in ollama_models:
            backends.append(
                {
                    "name": "ollama",
                    "kind": "ollama",
                    "base_url": args.ollama_url,
                    "model": model,
                }
            )

    if args.backend in ("mlx", "both"):
        if not mlx_models:
            raise ValueError(
                "No MLX models found. Use --mlx-model or run with --list-models to verify discovery."
            )
        for model in mlx_models:
            backends.append(
                {
                    "name": "mlx",
                    "kind": "openai",
                    "base_url": args.mlx_url,
                    "model": model,
                }
            )

    if not backends:
        raise ValueError("No backends configured. Provide at least one model for the selected backend.")

    return backends


def print_summary(results: List[RunResult]) -> None:
    def print_table(title: str, rows: List[RunResult]) -> None:
        by_backend: Dict[tuple[str, str], List[RunResult]] = {}
        for r in rows:
            by_backend.setdefault((r.backend, r.model), []).append(r)

        if not by_backend:
            print(f"\n=== {title} ===")
            print("(no rows)")
            return

        print(f"\n=== {title} ===")
        print("Time columns are in seconds (s). Throughput columns are tokens/second.")
        header_fmt = (
            "{:<12} {:<24} {:>6} {:>12} {:>12} {:>12} {:>16} {:>12} {:>12}"
        )
        row_fmt = (
            "{:<12} {:<24} {:>6} {:>12} {:>12} {:>12} {:>16} {:>12} {:>12}"
        )

        print(
            header_fmt.format(
                "backend",
                "model",
                "runs",
                "avg total",
                "p50% total",
                "avg ttft",
                "avg prompt tps",
                "avg gen tps",
                "avg load",
            )
        )
        print("-" * 126)

        for (backend, model), model_rows in by_backend.items():
            totals = [r.total_s for r in model_rows]
            avg_total = sum(totals) / len(model_rows)
            p50_total = statistics.median(totals)
            avg_ttft = mean_or_none([r.ttft_s for r in model_rows])
            avg_prompt_tps = mean_or_none([r.prompt_tps for r in model_rows])
            avg_gen_tps = mean_or_none([r.gen_tps for r in model_rows])
            avg_load_s = mean_or_none([r.load_s for r in model_rows])

            total_text = f"{avg_total:.3f}"
            p50_total_text = f"{p50_total:.3f}"
            ttft_text = f"{avg_ttft:.3f}" if avg_ttft is not None else "na"
            prompt_tps_text = f"{avg_prompt_tps:.1f}" if avg_prompt_tps is not None else "na"
            gen_tps_text = f"{avg_gen_tps:.1f}" if avg_gen_tps is not None else "na"
            load_s_text = f"{avg_load_s:.3f}" if avg_load_s is not None else "na"

            print(
                row_fmt.format(
                    backend,
                    model,
                    len(model_rows),
                    total_text,
                    p50_total_text,
                    ttft_text,
                    prompt_tps_text,
                    gen_tps_text,
                    load_s_text,
                )
            )

    warm_rows = [r for r in results if not r.cold]
    cold_rows = [r for r in results if r.cold]

    print_table("SUMMARY (WARM)", warm_rows)
    if cold_rows:
        print_table("SUMMARY (COLD)", cold_rows)


def bench_ollama(
    base_url: str,
    model: str,
    backend_name: str,
    prompt: str,
    prompt_index: int,
    cold: bool,
) -> RunResult:
    url = f"{base_url}/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": TEMPERATURE,
            "num_predict": MAX_TOKENS,
        },
    }

    start = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    data = resp.json()
    end = time.perf_counter()

    output = data.get("response", "")
    prompt_chars = len(prompt)
    output_chars = len(output)

    prompt_tokens = data.get("prompt_eval_count")
    output_tokens = data.get("eval_count")

    load_s = ((data.get("load_duration") or 0) / 1e9) if data.get("load_duration") is not None else None
    prompt_eval_s = ((data.get("prompt_eval_duration") or 0) / 1e9) if data.get("prompt_eval_duration") is not None else None
    eval_s = ((data.get("eval_duration") or 0) / 1e9) if data.get("eval_duration") is not None else None
    total_s = ((data.get("total_duration") or 0) / 1e9) if data.get("total_duration") is not None else (end - start)

    prompt_tps = (prompt_tokens / prompt_eval_s) if prompt_eval_s and prompt_tokens else None
    gen_tps = (output_tokens / eval_s) if eval_s and output_tokens else None
    ttft_s = (prompt_eval_s + (load_s or 0)) if prompt_eval_s is not None else None

    return RunResult(
        backend=backend_name,
        model=model,
        prompt_index=prompt_index,
        cold=cold,
        prompt_chars=prompt_chars,
        output_chars=output_chars,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_s=ttft_s,
        total_s=total_s,
        prompt_tps=prompt_tps,
        gen_tps=gen_tps,
        load_s=load_s,
        raw_meta=data,
    )


def bench_openai_compat(
    base_url: str,
    model: str,
    backend_name: str,
    prompt: str,
    prompt_index: int,
    cold: bool,
) -> RunResult:
    url = f"{base_url}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "stream": False,
    }

    headers = {"Content-Type": "application/json"}

    start = time.perf_counter()
    response = requests.post(url, headers=headers, json=payload, timeout=TIMEOUT)
    response.raise_for_status()
    data = response.json()
    end = time.perf_counter()

    message = data.get("choices", [{}])[0].get("message", {})
    output = message.get("content", "")

    usage = data.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens")
    output_tokens = usage.get("completion_tokens")

    prompt_chars = len(prompt)
    output_chars = len(output)
    total_s = end - start

    gen_tps = (output_tokens / total_s) if output_tokens and total_s > 0 else None

    return RunResult(
        backend=backend_name,
        model=model,
        prompt_index=prompt_index,
        cold=cold,
        prompt_chars=prompt_chars,
        output_chars=output_chars,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        ttft_s=None,
        total_s=total_s,
        prompt_tps=None,
        gen_tps=gen_tps,
        load_s=None,
        raw_meta={
            "output_preview": output[:120],
            "usage": usage,
        },
    )

def bench_backend(backend: Dict[str, str], prompt: str, prompt_index: int, cold: bool) -> RunResult:
    if backend["kind"] == "ollama":
        return bench_ollama(
            base_url=backend["base_url"],
            model=backend["model"],
            backend_name=backend["name"],
            prompt=prompt,
            prompt_index=prompt_index,
            cold=cold,
        )

    if backend["kind"] == "openai":
        return bench_openai_compat(
            base_url=backend["base_url"],
            model=backend["model"],
            backend_name=backend["name"],
            prompt=prompt,
            prompt_index=prompt_index,
            cold=cold,
        )

    raise ValueError(f"Unknown backend kind: {backend['kind']}")


def run_suite(backends: List[Dict[str, str]], warm_runs: int, include_cold: bool) -> List[RunResult]:
    results: List[RunResult] = []

    if include_cold:
        cold_prompt = PROMPTS[0]
        print("Cold runs...")
        for backend in backends:
            print(f"  starting cold: {backend['name']} | {backend['model']}", flush=True)
            results.append(bench_backend(backend, cold_prompt, prompt_index=0, cold=True))
            print(f"  finished cold: {backend['name']} | {backend['model']}", flush=True)

    print("Warm runs...")
    for rep in range(warm_runs):
        print(f"  repetition {rep + 1}/{warm_runs}", flush=True)
        for i, prompt in enumerate(PROMPTS):
            for backend in backends:
                print(
                    f"    starting: backend={backend['name']} model={backend['model']} prompt={i}",
                    flush=True,
                )
                results.append(bench_backend(backend, prompt, prompt_index=i, cold=False))
                print(
                    f"    finished: backend={backend['name']} model={backend['model']} prompt={i}",
                    flush=True,
                )

    return results


def main() -> None:
    args = parse_args()

    if args.list_models:
        print_discovered_models(args)
        return

    backends = build_backends(args)
    results = run_suite(backends, args.warm_runs, include_cold=args.include_cold)

    with open(args.output, "w", encoding="utf-8") as f:
        for row in results:
            f.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")

    print_summary(results)


if __name__ == "__main__":
    main()