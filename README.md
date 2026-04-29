
<h2 align="center">
  <img width="30%" alt="semble logo" src="https://raw.githubusercontent.com/MinishLab/semble/main/assets/images/semble_logo.png"><br/>
  Fast and Accurate Code Search for Agents
</h2>

<div align="center">
  <h2>
    <a href="https://pypi.org/project/semble/"><img src="https://img.shields.io/pypi/v/semble?color=%23007ec6&label=pypi%20package" alt="Package version"></a>
    <a href="https://app.codecov.io/gh/MinishLab/semble">
      <img src="https://codecov.io/gh/MinishLab/semble/graph/badge.svg?token=SZKRFKPPCG" alt="Codecov">
    </a>
    <a href="https://github.com/MinishLab/semble/blob/main/LICENSE">
      <img src="https://img.shields.io/badge/license-MIT-green" alt="License - MIT">
    </a>
  </h2>

[Quickstart](#quickstart) •
[Main Features](#main-features) •
[CLI](#cli) •
[OpenCode Tools](#opencode-tools) •
[How it works](#how-it-works) •
[Benchmarks](#benchmarks)

</div>

Semble is a code search library built for agents. It returns the exact code snippets they need instantly, cutting both token usage and waiting time on every step. Indexing and searching a full codebase end-to-end takes under a second, with ~200x faster indexing and ~10x faster queries than a code-specialized transformer, at 99% of its retrieval quality (see [benchmarks](#benchmarks)). Everything runs on CPU with no API keys, GPU, or external services. Use it from Python, the `semble` CLI, or OpenCode custom tools backed by a local on-disk cache.

## Quickstart

```bash
pip install semble  # Install with pip
uv add semble       # Install with uv
```

```python
from semble import SembleIndex

# Index a local directory
index = SembleIndex.from_path("./my-project")

# Index a remote git repository
index = SembleIndex.from_git("https://github.com/MinishLab/model2vec")

# Search the index with a natural-language or code query
results = index.search("save model to disk", top_k=3)

# Find code similar to a specific result
related = index.find_related(results[0], top_k=3)

# Each result exposes the matched chunk
result = results[0]
result.chunk.file_path   # "model2vec/model.py"
result.chunk.start_line  # 127
result.chunk.end_line    # 150
result.chunk.content     # "def save_pretrained(self, path: PathLike, ..."
```

## Main Features

- **Fast**: indexes a repo in ~250 ms and answers queries in ~1.5 ms, all on CPU.
- **Accurate**: NDCG@10 of 0.854 on our [benchmarks](#benchmarks), on par with code-specialized transformer models, at a fraction of the size and cost.
- **Local and remote**: pass a local path or a git URL.
- **Agent-ready CLI**: direct search and related-code commands with markdown snippets and persistent local cache.
- **Zero setup**: runs on CPU with no API keys, GPU, or external services required.

## CLI

Search a local path or git URL directly from the command line:

```bash
semble search "save model to disk" --path . --mode hybrid --top-k 5
semble related --file src/semble/index/index.py --line 181 --path . --top-k 5
semble related src/semble/index/index.py:181 --path . --top-k 5
```

Manage the on-disk cache:

```bash
semble cache stats --path .
semble cache clear --path .
```

`--path` defaults to the current working directory. Results are returned as markdown snippets with file paths, line ranges, scores, and fenced code blocks.

### Cache Behavior

Indexes are cached under `~/.cache/semble` by default. Override this with:

```bash
SEMBLE_CACHE_DIR=/custom/cache/path semble search "query" --path .
```

Each request checks whether the selected path is fresh before searching. For git worktrees, Semble fingerprints the worktree root using `HEAD`, `git status --porcelain`, and changed-file metadata. For non-git directories, it fingerprints included file paths, mtimes, and sizes. If the fingerprint changes, Semble rebuilds the index before returning results.

Multiple OpenCode or CLI instances can use the same cache. Semble uses per-index lock directories and atomic writes so concurrent processes do not read or write partial index entries. A second process waits for or reuses the first completed cache entry.

## OpenCode Tools

Install the Semble OpenCode custom tools globally so they are available in every OpenCode project:

- `~/.config/opencode/tools/code_search.ts` registers `code_search`.
- `~/.config/opencode/tools/code_search_related.ts` registers `code_search_related`.

The tools default to `context.worktree`, so the directory OpenCode was launched from is not the cache identity once the path is normalized. You can pass `path` explicitly to search another local path or git URL.

This repository includes reusable templates under `examples/opencode/`. Copy them into your global OpenCode tools directory on machines where you want Semble tools installed:

```bash
mkdir -p ~/.config/opencode/tools
cp examples/opencode/code_search.ts ~/.config/opencode/tools/code_search.ts
cp examples/opencode/code_search_related.ts ~/.config/opencode/tools/code_search_related.ts
```

If `semble` is not on `PATH`, set `SEMBLE_BIN` before launching OpenCode:

```bash
SEMBLE_BIN=/absolute/path/to/venv/bin/semble opencode
```

The wrapper executes `SEMBLE_BIN` as a single executable path. If you use `uv`, install Semble into an environment or create a small wrapper script rather than setting `SEMBLE_BIN` to a multi-word command.

## Local Development

Install this checkout in editable mode:

```bash
pip install -e /absolute/path/to/semble
uv pip install -e /absolute/path/to/semble
```

If OpenCode is using a globally installed `semble` executable, refresh it after CLI entrypoint changes:

```bash
uv tool install --force .
```

Then run:

```bash
semble search "search" --path /absolute/path/to/semble --top-k 3
```

To use a local editable install from OpenCode, launch it with the installed executable:

```bash
SEMBLE_BIN=/absolute/path/to/venv/bin/semble opencode
```

## How it works

Semble splits each file into code-aware chunks using [Chonkie](https://github.com/chonkie-inc/chonkie), then scores every query against the chunks with two complementary retrievers: static [Model2Vec](https://github.com/MinishLab/model2vec) embeddings using the code-specialized [potion-code-16M](https://huggingface.co/minishlab/potion-code-16M) model for semantic similarity, and [BM25](https://github.com/xhluca/bm25s) for lexical matches on identifiers and API names. The two score lists are fused with Reciprocal Rank Fusion (RRF).

After fusing, results are reranked with a set of code-aware signals:

<details>
<summary><b>Ranking signals</b></summary>

- **Adaptive weighting.** Symbol-like queries (`Foo::bar`, `_private`, `getUserById`) get more lexical weight, while natural-language queries stay balanced between semantic and lexical retrievers.
- **Definition boosts.** A chunk that defines the queried symbol (a `class`, `def`, `func`, etc.) is ranked above chunks that merely reference it.
- **Identifier stems.** Query tokens are stemmed and matched against identifier stems in a chunk, giving an additional weight to chunks that contain them. For example, querying `parse config` boosts chunks containing `parseConfig`, `ConfigParser`, or `config_parser`.
- **File coherence.** When multiple chunks from the same file match the query, the file is boosted so the top result reflects broad file-level relevance rather than a single out-of-context chunk.
- **Noise penalties.** Test files, `compat/`/`legacy/` shims, example code, and `.d.ts` declaration stubs are down-ranked so canonical implementations surface first.

</details>

Because the embedding model is static with no transformer forward pass at query time, all of this runs in milliseconds on CPU.

## Benchmarks

We benchmark quality and speed across all methods on ~1,250 queries over 63 repositories in 19 languages. The x-axis is total latency (index + first query); the y-axis is NDCG@10. Marker size reflects model parameter count.

![Speed vs quality](https://raw.githubusercontent.com/MinishLab/semble/main/assets/images/speed_vs_ndcg_cold.png)

| Method | NDCG@10 | Index time | Query p50 |
|--------|--------:|-----------:|----------:|
| CodeRankEmbed Hybrid | 0.862 | 57 s | 16 ms |
| **semble** | **0.854** | **263 ms** | **1.5 ms** |
| CodeRankEmbed | 0.765 | 57 s | 16 ms |
| ColGREP | 0.693 | 5.8 s | 124 ms |
| BM25 | 0.673 | 263 ms | 0.02 ms |
| grepai | 0.561 | 35 s | 48 ms |
| probe | 0.387 | — | 207 ms |
| ripgrep | 0.126 | — | 12 ms |

Semble achieves 99% of the performance of the 137M-parameter [CodeRankEmbed](https://huggingface.co/nomic-ai/CodeRankEmbed) Hybrid, while indexing 218x faster and answering queries 11x faster. See [benchmarks](benchmarks/README.md) for per-language results, ablations, and methodology.

## License

MIT

## Citing

If you use Semble in your research, please cite the following:

```bibtex
@software{minishlab2026semble,
  author       = {{van Dongen}, Thomas and Stephan Tulkens},
  title        = {Semble: Fast and Accurate Code Search for Agents},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.19785932},
  url          = {https://github.com/MinishLab/semble},
  license      = {MIT}
}
```
