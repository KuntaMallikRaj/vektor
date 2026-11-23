# Vektor

**A real-time vector database, built from scratch, for data that never stops changing.**

HNSW graph search · product quantization · filter-aware search · standing similarity queries over WebSockets.

> 🚧 Early development — Phase 0. See [VISION.md](VISION.md) for the why and [ROADMAP.md](ROADMAP.md) for the plan.

## Setup

```bash
uv sync
uv run pytest
```

## Layout

```
src/vektor/       the engine
tests/            unit tests (write the test, then make it pass)
benchmarks/       benchmark scripts and results
docs/journal.md   daily build log — raw material for blog posts
data/             datasets (git-ignored)
```
