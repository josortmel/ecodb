# EcoDB Benchmarks: Full Results

## LoCoMo (ACL 2024)

[LoCoMo](https://arxiv.org/abs/2402.17753) (Maharana et al., ACL 2024) is a long-context conversational memory benchmark. 10 conversations, 1,982 queries, session-level retrieval.

### With Docling chunking (5-turn sliding windows, 1-turn overlap)

This is EcoDB's standard ingestion pipeline. Docling parses conversational sessions into 5-turn sliding windows before embedding. Structured chunking is how every production memory system processes conversational data. EcoDB ships Docling as part of the system.

| Metric | K=5 | K=10 | K=20 |
|--------|:---:|:----:|:----:|
| **Recall@1** | 0.774 | 0.756 | 0.774 |
| **Recall@5** | 0.914 | 0.906 | **0.922** |
| **Recall@10** | * | 0.931 | **0.959** |

\* K=5 returns only 5 results, so R@10 cannot improve beyond R@5.

**By query type** (Recall@5, K=20): adversarial 0.946 (446 queries), open-domain 0.936 (841), temporal 0.916 (321), single-hop 0.911 (282), multi-hop 0.728 (92)

### Without chunking, with cross-encoder reranking (MiniLM-L-6-v2)

Raw session text, no sliding windows. Reranker enabled (stage 10 of GAMR pipeline).

|         | Baseline | With reranker | Delta   |
|---------|:--------:|:------------:|:-------:|
| R@1     | 0.414    | 0.578        | +0.164  |
| R@5     | 0.769    | 0.793        | +0.024  |
| R@10    | 0.894    | 0.869        | -0.025  |

The reranker significantly improves R@1 (finding the right answer at rank 1) without chunking. R@10 drops slightly because the reranker reshuffles candidates. The combination of chunking + reranker has not been evaluated yet.

### Key insight

Chunking alone improved R@5 from 0.769 to 0.922 (+19.9%). Both measured at K=20. The reranker improved R@1 from 0.414 to 0.578 (+39.6%). For conversational data, ingestion granularity matters more than ranking sophistication for R@5, but ranking sophistication matters for R@1.

---

## Internal golden set (paragraph-level)

A custom benchmark against EcoDB's production corpus. Harder than LoCoMo by design: paragraph-level retrieval (find a specific memory, not just the right session) in a multi-language, multi-topic production corpus.

- 1,400+ memories
- Multiple languages (Spanish, English, mixed)
- Dozens of distinct topics (technical architecture, personal reflections, meeting notes, decisions, image memories)
- 60+ queries with known target memories
- No curation, no cherry-picking

### Global results

| Metric | Score |
|--------|:-----:|
| **Recall@5** | **0.56** |
| **MRR** | **0.39** |
| **Multimodal R@5** | **0.70** |

### By category (golden set v2, 120 queries, 3 annotators)

| Category | R@5 |
|----------|:---:|
| analytical | 0.593 |
| contextual | 0.593 |
| factual | 0.485 |
| historical | 0.455 |
| cross_modal | 0.273 |
| **Global** | **0.508** |

Note: the v2 golden set (120 queries, 3 annotators) scores lower than v1 (60 queries, 1 annotator) because more annotators and more queries reduce single-annotator bias. The v2 numbers are more reliable.

### Why we maintain this benchmark

Standard benchmarks like LoCoMo measure session-level retrieval: find which conversation contains the answer. That's finding a book on a shelf. Our golden set measures paragraph-level retrieval: find the specific memory in the library. This is where we explore our margin of improvement. It's the benchmark that still challenges the system.

The AI memory space has a methodology problem: leaderboard numbers that look impressive on curated benchmarks but don't survive contact with real-world heterogeneous data. We publish both LoCoMo (standardized, comparable) and our golden set (harder, honest) because we believe transparency about what works and what doesn't is more valuable than a clean leaderboard.

---

## Methodology

All evaluation scripts are in [`eval/`](eval/). Results are reproducible. No conversations or queries excluded.
