import argparse
import asyncio
import json
import random
import subprocess
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import tiktoken
from model2vec import StaticModel
from openai import AsyncOpenAI

from benchmarks.data import (
    RepoSpec,
    Target,
    Task,
    add_filter_args,
    grouped_tasks,
    load_filtered_tasks,
    results_path,
    save_results,
    target_matches_location,
)
from semble import SembleIndex
from semble.index.dense import _DEFAULT_MODEL_NAME
from semble.index.file_walker import DEFAULT_IGNORED_DIRS, FILE_TYPES, FileCategory
from semble.types import Chunk

_RG_INCLUDE_GLOBS: tuple[str, ...] = tuple(
    f"*{ext}" for ext, spec in FILE_TYPES.items() if spec.category == FileCategory.CODE
)
_RG_EXCLUDE_GLOBS: tuple[str, ...] = tuple(f"!{d}" for d in DEFAULT_IGNORED_DIRS)

_BUDGETS = (500, 1000, 2000, 4000, 8000, 16000, 32000)
_PLOT_BUDGETS = sorted({int(b) for b in np.logspace(np.log10(100), np.log10(64000), 60)})
_TOKENIZER_NAME = "cl100k_base"
_RG_CONTEXTS = (8,)
_RG_MAX_MATCHES = 500
_SEMBLE_TOP_K = 50
_JUDGE_MODEL = "gpt-5-mini"
_JUDGE_CONCURRENCY = 8
_JUDGE_RETRIES = 3
_JUDGE_DEFAULT_SAMPLE = 200
_JUDGE_TOP_KS = (3, 5, 10, 20)
_JUDGE_CONTEXT_CAP = 32_000

_IMAGES_DIR = Path(__file__).parent.parent / "assets" / "images"
_RESULTS_DIR = Path(__file__).parent / "results"

_JUDGE_PROMPT = """\
You are evaluating whether retrieved code context is sufficient to answer a code search query.

A code agent ran a search for the query below and got back the retrieved context shown. \
Decide whether that context contains enough relevant code for an engineer to give a substantive \
answer to the query (showing how something is implemented, where it lives, or how it works).

Reply YES if the context contains code that directly addresses the query.
Reply NO if the context is empty, off-topic, or only tangentially related.

Reply with exactly one word: YES or NO.

Query: {query}

Retrieved context:
{context}
"""


Curve: TypeAlias = list[tuple[int, int]]
MethodCurves: TypeAlias = list[tuple[Curve, int]]


@dataclass(frozen=True)
class JudgeRecord:
    """One LLM-judge sufficiency verdict."""

    repo: str
    query: str
    category: str
    method: str
    tokens_used: int
    answered: bool


def _semble_units(index: SembleIndex, query: str) -> list[Chunk]:
    """Top-K semble chunks as units, ordered by score."""
    return [r.chunk for r in index.search(query, top_k=_SEMBLE_TOP_K)]


def _rg_command(pattern: str, repo_dir: Path) -> list[str]:
    """Build the rg command line, scoped to the same code-file universe semble indexes."""
    cmd = ["rg", "--json", "--fixed-strings", "--ignore-case"]
    for glob in (*_RG_EXCLUDE_GLOBS, *_RG_INCLUDE_GLOBS):
        cmd += ["--glob", glob]
    cmd += [pattern, str(repo_dir)]
    return cmd


def _rg_matches(pattern: str, repo_dir: Path) -> list[tuple[str, int]]:
    """Return (file_path, line_number) matches via rg --json, in rg's output order."""
    try:
        proc = subprocess.run(
            _rg_command(pattern, repo_dir),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return []
    if proc.returncode not in (0, 1):
        return []
    matches: list[tuple[str, int]] = []
    for line in proc.stdout.splitlines():
        if not line:
            continue
        try:
            evt = json.loads(line)
        except json.JSONDecodeError:
            continue
        if evt.get("type") != "match":
            continue
        data = evt.get("data", {})
        path = data.get("path", {}).get("text")
        ln = data.get("line_number")
        if path and isinstance(ln, int):
            matches.append((path, ln))
    return matches


def _ripgrep_units(query: str, repo_dir: Path, context: int) -> list[Chunk]:
    """Ripgrep windows: files ranked by match count desc, windows merged within file."""
    matches = _rg_matches(query, repo_dir)
    if not matches:
        return []
    per_file: dict[str, list[int]] = defaultdict(list)
    for path, ln in matches[:_RG_MAX_MATCHES]:
        per_file[path].append(ln)
    ranked = sorted(per_file.items(), key=lambda kv: (-len(kv[1]), kv[0]))
    units: list[Chunk] = []
    for path, lines in ranked:
        windows: list[list[int]] = []
        for ln in sorted(set(lines)):
            start, end = max(1, ln - context), ln + context
            if windows and start <= windows[-1][1] + 1:
                windows[-1][1] = max(windows[-1][1], end)
            else:
                windows.append([start, end])
        try:
            file_lines = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for start, end in windows:
            end_clamped = min(end, len(file_lines))
            text = "\n".join(file_lines[start - 1 : end_clamped])
            units.append(Chunk(content=text, file_path=path, start_line=start, end_line=end_clamped))
    return units


def _grep_file_units(
    pattern: str,
    repo_dir: Path,
) -> list[Chunk]:
    """Return whole matched files in match-count order."""
    matches = _rg_matches(pattern, repo_dir)
    if not matches:
        return []
    ranked = sorted(Counter(path for path, _ in matches[:_RG_MAX_MATCHES]).items(), key=lambda kv: (-kv[1], kv[0]))
    units: list[Chunk] = []
    for path, _ in ranked:
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        units.append(Chunk(content=text, file_path=path, start_line=1, end_line=text.count("\n") + 1))
    return units


def _retrieval_units_for_task(
    index: SembleIndex,
    task: Task,
    repo_dir: Path,
    *,
    include_ripgrep: bool = False,
) -> list[tuple[str, list[Chunk]]]:
    """Return retrieval-method units for a task in benchmark order."""
    methods = [
        ("semble", _semble_units(index, task.query)),
        ("grep+read", _grep_file_units(task.query, repo_dir)),
    ]
    if include_ripgrep:
        methods.extend((f"ripgrep-c{c}", _ripgrep_units(task.query, repo_dir, c)) for c in _RG_CONTEXTS)
    return methods


def _retrieval_units_for_judge(
    index: SembleIndex,
    task: Task,
    repo_dir: Path,
) -> list[tuple[str, list[Chunk]]]:
    """Return (method, units) for the judge: semble sliced at each top_k, plus grep+read."""
    all_semble = _semble_units(index, task.query)
    methods: list[tuple[str, list[Chunk]]] = [(f"semble-top{k}", all_semble[:k]) for k in _JUDGE_TOP_KS]
    methods.append(("grep+read", _grep_file_units(task.query, repo_dir)))
    return methods


def _curve(units: list[Chunk], targets: tuple[Target, ...], enc: Any) -> Curve:
    """Cumulative (tokens, covered_target_count) after each retrieved unit, starting at (0, 0)."""
    covered = [False] * len(targets)
    cumulative = 0
    points: Curve = [(0, 0)]
    for unit in units:
        cumulative += len(enc.encode(unit.content, disallowed_special=()))
        for i, tgt in enumerate(targets):
            if not covered[i] and target_matches_location(unit.file_path, unit.start_line, unit.end_line, tgt):
                covered[i] = True
        points.append((cumulative, sum(covered)))
    return points


def _recall_at(curve: Curve, budget: int, n_total: int) -> float:
    """Recall (covered_targets / n_total) at the largest cumulative-tokens point <= budget."""
    if n_total == 0 or not curve:
        return 0.0
    covered_at = 0
    for tokens, covered in curve:
        if tokens > budget:
            break
        covered_at = covered
    return covered_at / n_total


def _mean_recall_at(curves: MethodCurves, budgets: tuple[int, ...] | list[int]) -> dict[int, float]:
    """Mean recall at each budget across all queries."""
    return {b: float(np.mean([_recall_at(c, b, n) for c, n in curves if n > 0])) if curves else 0.0 for b in budgets}


def _mean_curve(curves: MethodCurves, grid: list[int]) -> list[float]:
    """Mean recall over all queries at each grid budget."""
    samples = [[_recall_at(c, b, n) for b in grid] for c, n in curves if n > 0]
    return np.mean(samples, axis=0).tolist() if samples else [0.0] * len(grid)


def _tokens_to_first_hit(curve: Curve) -> int | None:
    """Cumulative tokens at which the first relevant target is covered, or None if never."""
    for tokens, covered in curve:
        if covered > 0:
            return tokens
    return None


def _pairwise_reduction(semble: MethodCurves, other: MethodCurves) -> dict[str, float] | None:
    """Median 'tokens to first hit' reduction, paired on queries where both methods hit."""
    pairs: list[tuple[int, int]] = []
    for (s_curve, _), (o_curve, _) in zip(semble, other, strict=True):
        s = _tokens_to_first_hit(s_curve)
        o = _tokens_to_first_hit(o_curve)
        if s is not None and o is not None and o > 0:
            pairs.append((s, o))
    if not pairs:
        return None
    ratios = [s / o for s, o in pairs]
    return {
        "n_paired": float(len(pairs)),
        "median_semble_tokens": float(np.median([s for s, _ in pairs])),
        "median_other_tokens": float(np.median([o for _, o in pairs])),
        "median_reduction": 1.0 - float(np.median(ratios)),
        "mean_reduction": 1.0 - float(np.mean(ratios)),
        "semble_better_pct": sum(1 for s, o in pairs if s < o) / len(pairs),
    }


def _format_context(units: list[Chunk], budget: int, enc: Any) -> tuple[str, int]:
    """
    Concatenate units with file headers, hard-capped at budget tokens.

    Greedily appends blocks until the running estimate exceeds budget, then re-encodes the
    joined string and slices by token IDs as the final guarantee.
    """
    if not units or budget <= 0:
        return "(no context retrieved)", 0

    parts: list[str] = []
    rough_tokens = 0
    for unit in units:
        block = f"// {unit.location}\n{unit.content}"
        parts.append(block)
        rough_tokens += len(enc.encode(block, disallowed_special=())) + 2  # +2 for "\n\n"
        if rough_tokens >= budget + 64:
            break

    context = "\n\n".join(parts)
    ids = enc.encode(context, disallowed_special=())
    if len(ids) > budget:
        marker = "\n... [truncated]"
        marker_ids = enc.encode(marker, disallowed_special=())
        keep = max(0, budget - len(marker_ids))
        context = enc.decode(ids[:keep]) + marker
        ids = enc.encode(context, disallowed_special=())
        if len(ids) > budget:
            context = enc.decode(ids[:budget])
            ids = enc.encode(context, disallowed_special=())
    if len(ids) > budget:
        raise AssertionError(f"context formatting exceeded budget: {len(ids)} > {budget}")
    return context, len(ids)


async def _judge_one(client: AsyncOpenAI, query: str, context: str) -> bool | None:
    """Call the judge model; return True/False or None on parse failure or repeated errors."""
    prompt = _JUDGE_PROMPT.format(query=query, context=context)
    last_err: Exception | None = None
    for attempt in range(_JUDGE_RETRIES):
        try:
            resp = await client.responses.create(
                model=_JUDGE_MODEL,
                input=prompt,
                reasoning={"effort": "minimal"},
            )
            text = (resp.output_text or "").strip().upper()
            if text.startswith("Y"):
                return True
            if text.startswith("N"):
                return False
            return None
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            await asyncio.sleep(1.5 * (attempt + 1))
    print(f"  judge failed after {_JUDGE_RETRIES} retries: {last_err}", file=sys.stderr)
    return None


async def _judge_many(pending: list[tuple[Task, str, str, int]], concurrency: int) -> list[JudgeRecord]:
    """Run judge calls with bounded concurrency."""
    client = AsyncOpenAI()
    sem = asyncio.Semaphore(concurrency)
    results: list[JudgeRecord | None] = [None] * len(pending)

    async def worker(i: int, task: Task, method: str, context: str, tokens: int) -> None:
        async with sem:
            verdict = await _judge_one(client, task.query, context)
            if verdict is None:
                return
            results[i] = JudgeRecord(
                repo=task.repo,
                query=task.query,
                category=task.category,
                method=method,
                tokens_used=tokens,
                answered=verdict,
            )

    await asyncio.gather(*(worker(i, *p) for i, p in enumerate(pending)))
    return [r for r in results if r is not None]


def _stratified_sample(tasks: list[Task], n: int, seed: int) -> list[Task]:
    """Sample n tasks stratified across categories."""
    rng = random.Random(seed)
    by_cat: dict[str, list[Task]] = defaultdict(list)
    for task in tasks:
        by_cat[task.category].append(task)
    cats = sorted(by_cat)
    per_cat = n // len(cats)
    picked: list[Task] = []
    for cat in cats:
        bucket = by_cat[cat]
        rng.shuffle(bucket)
        picked.extend(bucket[:per_cat])
    rng.shuffle(picked)
    return picked


def _evaluate_repo_recall(
    index: SembleIndex,
    tasks: list[Task],
    repo_dir: Path,
    enc: Any,
) -> dict[str, MethodCurves]:
    """Build per-method curves for every task in the repo."""
    methods: dict[str, MethodCurves] = defaultdict(list)

    for task in tasks:
        targets = task.all_relevant
        n = len(targets)
        for method, units in _retrieval_units_for_task(
            index,
            task,
            repo_dir,
            include_ripgrep=True,
        ):
            methods[method].append((_curve(units, targets, enc), n))
    return dict(methods)


def _build_pending_judge(
    tasks: list[Task],
    repo_specs: dict[str, RepoSpec],
    model: StaticModel,
    enc: Any,
) -> list[tuple[Task, str, str, int]]:
    """Build (task, method, context, tokens) inputs for the LLM judge across all sampled tasks."""
    pending: list[tuple[Task, str, str, int]] = []
    print(f"\n{'Repo':<22} {'Tasks':>6} {'Time':>8}", file=sys.stderr)
    print(f"{'-' * 22} {'-' * 6} {'-' * 8}", file=sys.stderr)
    for repo, task_list in sorted(grouped_tasks(tasks).items()):
        spec = repo_specs[repo]
        started = time.perf_counter()
        index = SembleIndex.from_path(spec.benchmark_dir, model=model)
        for task in task_list:
            for method, units in _retrieval_units_for_judge(index, task, spec.benchmark_dir):
                context, tokens = _format_context(units, _JUDGE_CONTEXT_CAP, enc)
                pending.append((task, method, context, tokens))
        print(f"{repo:<22} {len(task_list):>6} {time.perf_counter() - started:>7.1f}s", file=sys.stderr)
    return pending


def _aggregate_judge(records: list[JudgeRecord], attempted_per_method: dict[str, int]) -> dict[str, dict[str, Any]]:
    """Group records by method and compute answer rate, mean tokens, missing-verdict count."""
    by_method: dict[str, list[JudgeRecord]] = defaultdict(list)
    for r in records:
        by_method[r.method].append(r)
    out: dict[str, dict[str, Any]] = {}
    for method, attempted in attempted_per_method.items():
        recs = by_method.get(method, [])
        n_yes = sum(1 for r in recs if r.answered)
        by_cat: dict[str, list[bool]] = defaultdict(list)
        for r in recs:
            by_cat[r.category].append(r.answered)
        out[method] = {
            "n": len(recs),
            "n_attempted": attempted,
            "n_failed": attempted - len(recs),
            "answer_rate": n_yes / len(recs) if recs else 0.0,
            "mean_tokens_retrieved": float(np.mean([r.tokens_used for r in recs])) if recs else 0.0,
            "answer_rate_by_category": {c: sum(v) / len(v) for c, v in sorted(by_cat.items())},
        }
    return out


def _save_judge_records(records: list[JudgeRecord]) -> Path:
    """Write per-query judge records as compact JSONL for version comparisons."""
    out = results_path("context-efficiency-judge-records").with_suffix(".jsonl")
    lines = [
        json.dumps(
            {
                "repo": r.repo,
                "query": r.query,
                "category": r.category,
                "method": r.method,
                "tokens": r.tokens_used,
                "answered": r.answered,
            },
            separators=(",", ":"),
        )
        for r in records
    ]
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


_PLOT_STYLE: dict[str, dict[str, object]] = {
    "semble": {"label": "semble", "color": "#1a5fa8", "linewidth": 2.4, "zorder": 4},
    "grep+read": {"label": "grep + read file", "color": "#922b21", "linewidth": 1.8, "zorder": 3},
    "ripgrep-c8": {
        "label": "ripgrep -C 8 (snippets)",
        "color": "#707070",
        "linewidth": 1.4,
        "zorder": 2,
        "linestyle": "--",
    },
}


def _plot_recall_vs_tokens(payload: dict[str, Any], out_path: Path) -> None:
    """Render a recall-vs-tokens curve from a recall-mode payload."""
    plot_data = payload["plot"]
    budgets = plot_data["budgets"]
    recalls = plot_data["recall"]
    n_queries = payload.get("n_queries", "?")

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.grid(axis="y", color="#e8e8e8", linewidth=0.7, zorder=0)
    ax.grid(axis="x", color="#f0f0f0", linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#cccccc")

    for method, style in _PLOT_STYLE.items():
        if method not in recalls:
            continue
        ax.plot(
            budgets,
            recalls[method],
            label=style["label"],
            color=style["color"],
            linewidth=style["linewidth"],
            linestyle=style.get("linestyle", "-"),
            zorder=style["zorder"],
        )

    ax.set_xscale("log")
    ax.set_xlim(min(budgets), max(budgets))
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Retrieved context tokens", fontsize=10, color="#444444")
    ax.set_ylabel("Recall (relevant files surfaced)", fontsize=10, color="#444444")
    ax.set_title(
        f"Context efficiency: recall vs. retrieved tokens  (n={n_queries} queries)",
        fontsize=12,
        color="#222222",
        pad=12,
    )
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v / 1000:.0f}k" if v >= 1000 else f"{v:.0f}"))
    ax.tick_params(labelsize=9, colors="#555555")
    ax.legend(loc="lower right", fontsize=9, frameon=True, framealpha=0.95, edgecolor="#dddddd")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved plot to {out_path}", file=sys.stderr)


def _print_recall_summary(method_curves: dict[str, MethodCurves]) -> dict[str, dict[int, float]]:
    """Print and return recall@budget per method."""
    print(f"\nTokenizer: {_TOKENIZER_NAME}\n", file=sys.stderr)
    print(f"{'Method':<20} " + "  ".join(f"{b:>7}" for b in _BUDGETS), file=sys.stderr)
    print(f"{'-' * 20} " + "  ".join(f"{'-' * 7:>7}" for _ in _BUDGETS), file=sys.stderr)
    summary: dict[str, dict[int, float]] = {}
    for method, curves in method_curves.items():
        recall = _mean_recall_at(curves, _BUDGETS)
        summary[method] = recall
        print(f"{method:<20} " + "  ".join(f"{recall[b]:>7.3f}" for b in _BUDGETS), file=sys.stderr)
    return summary


def _print_first_hit_summary(method_curves: dict[str, MethodCurves]) -> dict[str, dict[str, float]]:
    """Print and return pairwise tokens-to-first-hit reductions vs semble."""
    print("\nTokens to first relevant file (semble vs other, paired)", file=sys.stderr)
    print(f"{'vs':<20} {'n':>5}  {'med-semble':>10}  {'med-other':>10}  {'med-reduce':>10}", file=sys.stderr)
    print(f"{'-' * 20} {'-' * 5}  {'-' * 10}  {'-' * 10}  {'-' * 10}", file=sys.stderr)
    reductions: dict[str, dict[str, float]] = {}
    for method, curves in method_curves.items():
        if method == "semble":
            continue
        red = _pairwise_reduction(method_curves["semble"], curves)
        if red is None:
            continue
        reductions[method] = red
        print(
            f"{method:<20} {int(red['n_paired']):>5}  "
            f"{red['median_semble_tokens']:>10.0f}  {red['median_other_tokens']:>10.0f}  "
            f"{red['median_reduction']:>10.1%}",
            file=sys.stderr,
        )
    return reductions


def run_recall(args: argparse.Namespace) -> None:
    """Run the recall-vs-token-budget benchmark."""
    repo_specs, tasks = load_filtered_tasks(args.repo or None, args.language or None)

    print("Loading tokenizer + model...", file=sys.stderr)
    enc = tiktoken.get_encoding(_TOKENIZER_NAME)
    model = StaticModel.from_pretrained(_DEFAULT_MODEL_NAME)

    method_curves: dict[str, MethodCurves] = defaultdict(list)
    print(f"\n{'Repo':<22} {'Language':<12} {'Tasks':>6} {'Time':>8}", file=sys.stderr)
    print(f"{'-' * 22} {'-' * 12} {'-' * 6} {'-' * 8}", file=sys.stderr)
    for repo, repo_task_list in sorted(grouped_tasks(tasks).items()):
        spec = repo_specs[repo]
        started = time.perf_counter()
        index = SembleIndex.from_path(spec.benchmark_dir, model=model)
        per_method = _evaluate_repo_recall(index, repo_task_list, spec.benchmark_dir, enc)
        for m, lst in per_method.items():
            method_curves[m].extend(lst)
        print(
            f"{repo:<22} {spec.language:<12} {len(repo_task_list):>6} {time.perf_counter() - started:>7.1f}s",
            file=sys.stderr,
        )

    summary = _print_recall_summary(method_curves)
    reductions = _print_first_hit_summary(method_curves)

    payload: dict[str, Any] = {
        "tool": "context-efficiency-recall",
        "tokenizer": _TOKENIZER_NAME,
        "budgets": list(_BUDGETS),
        "n_queries": len(method_curves["semble"]),
        "recall_at_budget": {m: {str(b): round(v, 4) for b, v in r.items()} for m, r in summary.items()},
        "first_hit_reduction": {m: {k: round(v, 4) for k, v in d.items()} for m, d in reductions.items()},
        "plot": {
            "budgets": _PLOT_BUDGETS,
            "recall": {m: [round(x, 4) for x in _mean_curve(c, _PLOT_BUDGETS)] for m, c in method_curves.items()},
        },
    }
    if args.repo or args.language:
        print(json.dumps(payload, indent=2))
        return

    out = save_results("context-efficiency-recall", payload)
    print(f"\nResults saved to {out}", file=sys.stderr)
    if not args.no_plot:
        _IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        _plot_recall_vs_tokens(payload, _IMAGES_DIR / "recall_vs_tokens.png")


def run_judge(args: argparse.Namespace) -> None:
    """Run the LLM-as-judge sufficiency benchmark on a stratified sample."""
    repo_specs, tasks = load_filtered_tasks()
    sample = _stratified_sample(tasks, args.sample, args.seed)
    print(f"Sampled {len(sample)} queries (seed={args.seed})", file=sys.stderr)

    print("Loading model + tokenizer...", file=sys.stderr)
    enc = tiktoken.get_encoding(_TOKENIZER_NAME)
    model = StaticModel.from_pretrained(_DEFAULT_MODEL_NAME)

    pending = _build_pending_judge(sample, repo_specs, model, enc)
    attempted_per_method: dict[str, int] = defaultdict(int)
    for _, method, _, _ in pending:
        attempted_per_method[method] += 1

    print(f"\nJudging {len(pending)} (query, method) pairs with {_JUDGE_MODEL}...", file=sys.stderr)
    started = time.perf_counter()
    records = asyncio.run(_judge_many(pending, args.concurrency))
    elapsed = time.perf_counter() - started
    print(f"  done in {elapsed:.1f}s ({len(records)}/{len(pending)} returned a verdict)", file=sys.stderr)
    n_missing = len(pending) - len(records)
    if n_missing:
        print(f"  WARNING: {n_missing} verdicts missing — see n_failed per method.", file=sys.stderr)

    summary = _aggregate_judge(records, dict(attempted_per_method))
    print(f"\n{'Method':<24} {'mean_tokens':>12} {'answer_rate':>12}", file=sys.stderr)
    print(f"{'-' * 24} {'-' * 12} {'-' * 12}", file=sys.stderr)
    for method, info in summary.items():
        print(
            f"{method:<24} {info['mean_tokens_retrieved']:>12.0f} {info['answer_rate']:>12.3f}",
            file=sys.stderr,
        )

    payload = {
        "tool": "context-efficiency-judge",
        "judge_model": _JUDGE_MODEL,
        "tokenizer": _TOKENIZER_NAME,
        "sample_size": len(sample),
        "seed": args.seed,
        "top_ks": list(_JUDGE_TOP_KS),
        "context_cap": _JUDGE_CONTEXT_CAP,
        "summary": summary,
    }
    records_out = _save_judge_records(records)
    payload["records_file"] = records_out.name
    out = save_results("context-efficiency-judge", payload)
    print(f"\nResults saved to {out}", file=sys.stderr)
    print(f"Records saved to {records_out}", file=sys.stderr)


def run_plot(args: argparse.Namespace) -> None:
    """Plot recall-vs-tokens from a saved recall-mode payload."""
    matches = sorted(_RESULTS_DIR.glob("context-efficiency-recall-*.json"))
    in_path = args.input or (matches[-1] if matches else None)
    if in_path is None:
        raise SystemExit(f"No recall results found in {_RESULTS_DIR}")
    payload = json.loads(in_path.read_text(encoding="utf-8"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _plot_recall_vs_tokens(payload, args.output)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Context-efficiency benchmark: semble vs grep workflows.")
    parser.set_defaults(func=run_recall, repo=[], language=[], no_plot=False)
    sub = parser.add_subparsers(dest="mode", required=False)

    recall = sub.add_parser("recall", help="Recall vs. token-budget across all queries (default).")
    add_filter_args(recall)
    recall.add_argument("--no-plot", action="store_true", help="Skip plotting after the run.")
    recall.set_defaults(func=run_recall)

    judge = sub.add_parser("judge", help="LLM-as-judge sufficiency on a stratified sample.")
    judge.add_argument("--sample", type=int, default=_JUDGE_DEFAULT_SAMPLE, help="Number of stratified queries.")
    judge.add_argument("--seed", type=int, default=42, help="Random seed for sampling.")
    judge.add_argument("--concurrency", type=int, default=_JUDGE_CONCURRENCY, help="Concurrent judge calls.")
    judge.set_defaults(func=run_judge)

    plot = sub.add_parser("plot", help="Regenerate the recall-vs-tokens plot from a saved JSON.")
    plot.add_argument("--input", type=Path, default=None, help="Path to recall results (default: newest).")
    plot.add_argument("--output", type=Path, default=_IMAGES_DIR / "recall_vs_tokens.png", help="Output PNG path.")
    plot.set_defaults(func=run_plot)

    return parser.parse_args()


def main() -> None:
    """Dispatch to the requested mode."""
    args = _parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
