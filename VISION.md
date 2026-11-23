# Vektor — Vision

> A real-time vector database, built from scratch, for data that never stops changing.

Every feature we build is checked against this document. If it doesn't serve the thesis, it waits.

---

## 1. What we're building

Vektor stores **embeddings** (vectors that represent meaning) and finds the closest ones fast.
Unlike most vector databases, which are tuned for a static dataset built once, Vektor targets
**streaming data**:

- **Constant churn** — vectors are inserted and deleted continuously, and recall must not decay.
- **Filtered search** — "similar to X, but only last hour and severity = high" must not collapse recall.
- **Standing queries** — clients subscribe to a query and get a **WebSocket push** the moment a
  matching vector arrives.

```
 producers ──insert/delete──▶ ┌──────────────────────────┐ ──search──▶ clients
                              │  HNSW + PQ index         │
                              │  filter-aware search     │
                              │  standing-query matcher  │ ──WebSocket push──▶ subscribers
                              │  WAL + snapshots         │
                              └──────────────────────────┘
```

**Flagship use case — real-time threat detection:** embed every incoming security log line and
alert instantly when it is close to a known attack pattern.

## 2. Why we're doing it

- Move into **infra / AI-infra** roles at top-tier companies.
- A **DSA-heavy** project: graphs, heaps, skip-list layering, k-means, quantization.
- Every layer built by us — so every line can be **defended in an interview**.
- **Numbers on the resume** (recall, QPS, latency, memory), not a feature list.

## 3. Why the need exists

- Vector search powers search-by-meaning, RAG/LLM apps, recommendations, and security analytics.
- Real data streams. Graph indexes like HNSW are known to **lose recall under heavy deletes/updates**;
  most systems fall back to periodic rebuilds.
- **Filtered ANN is still an open problem** — restrictive filters make graph search wander
  (see Filtered-DiskANN, ACORN).
- **Standing similarity queries** ("percolator for vectors") are rare.

## 4. How we meet it

| Need | Approach | Core DSA |
|---|---|---|
| Fast search | HNSW from scratch | Layered graph, greedy/best-first search, heaps |
| Memory efficiency | Product Quantization, IVF | K-means, lookup tables |
| Churn without decay | Tombstones, local graph repair, consolidation | Graph repair, priority queues |
| Filters that don't collapse | Filter-aware traversal, selectivity-based strategy | Bitmaps, cost estimation |
| Standing queries | Index the queries; match each new vector | Reverse kNN, range search |
| Durability | WAL, snapshots, crash recovery | Append-only logs |
| Real-time push | Own WebSocket layer | Ring buffers, backpressure |
| Scale | Sharding + scatter-gather | K-way heap merge |

## 5. How it stands out

- **A thesis**, not a tutorial: *vector search for streaming data*.
- **Benchmarks others don't publish:** recall under 24h of churn; recall vs. filter selectivity
  (50% → 0.1%); standing-query notification latency.
- **One distinctive feature:** standing similarity queries over WebSockets.
- **Engineering discipline:** benchmark harness, crash-recovery tests, design docs, write-ups of
  bugs we found.

## 6. How we define winning

We do **not** claim to beat Qdrant / Milvus / FAISS on general speed. We win on claims we can defend:

- [ ] Recall@10 stays ≥ 95% after 10M updates with no full rebuild
- [ ] At 0.1% filter selectivity, recall is X% higher than naive post-filtering
- [ ] Standing-query subscribers notified within Y ms (p99) of a matching insert
- [ ] Within a published factor of hnswlib on SIFT-1M (recall vs. QPS curve)

X and Y get filled in with real measurements — never guesses.

**Week-1 validation task:** survey what Qdrant, Weaviate, Milvus and recent papers already do for
churn, filtering, and vector percolation. If the gap is smaller than we think, we sharpen the angle.

## 7. Ground rules

- Python + NumPy for math. **Every index algorithm is ours.**
- FAISS / hnswlib appear **only as benchmark baselines**, never inside the engine.
- Every phase ends with: working code + a benchmark graph + a short write-up in `docs/`.
- **The core algorithms are written by the project owner** (HNSW, PQ, filtering, WebSocket layer).
  Claude explains papers, reviews hard, builds test/benchmark scaffolding, and challenges designs.
