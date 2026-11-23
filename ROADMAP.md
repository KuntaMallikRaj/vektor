# Vektor — Roadmap

~1 hour/day. Each phase ends with working code + a benchmark graph + a write-up in `docs/`.

## Phase 0 — Foundations & benchmark harness (weeks 1–2)  ← we are here
- [ ] Distance functions: L2, cosine, dot (vectorized NumPy)
- [ ] Brute-force exact kNN with top-k selection — the ground truth
- [ ] recall@k metric
- [ ] Load SIFT-1M (`.fvecs` / `.ivecs` readers)
- [ ] Benchmark script: recall, QPS, p50/p99 latency
- [ ] Week-1 research: what do Qdrant / Weaviate / Milvus do for churn, filtering, percolation?

## Phase 1 — HNSW from scratch (weeks 3–8)
- Greedy graph search → multi-layer HNSW (layer assignment, insert, neighbor heuristic)
- Tune M / ef_construction / ef_search; recall-vs-QPS curve
- Deletes (tombstones), save/load, benchmark vs hnswlib, profile
- 📄 Malkov & Yashunin, *Efficient and robust ANN search using HNSW graphs* (2016)

## Phase 2 — Quantization (weeks 9–12)
- K-means from scratch → Product Quantization → ADC lookup tables → IVF-PQ + re-ranking
- 📄 Jégou et al., *Product Quantization for Nearest Neighbor Search* (2011)

## Phase 3 — Churn, filtering, durability (weeks 13–16)
- Churn benchmark (recall over continuous insert/delete), graph repair
- Filter-aware search; recall vs. selectivity benchmark
- WAL, mmap, snapshots, crash recovery
- 📄 DiskANN / FreshDiskANN / Filtered-DiskANN / ACORN

## Phase 4 — Server + standing queries (weeks 17–20)
- HTTP API, batching, concurrent reads
- Own WebSocket layer; standing similarity queries pushed to subscribers

## Phase 5 — Distribution (weeks 21–24)
- Sharding, scatter-gather with k-way heap merge, replication, failure handling

## Phase 6 — Performance & launch (weeks 25–28)
- Rust (PyO3) SIMD distance kernels
- Full benchmark report, README graphs, launch blog post
