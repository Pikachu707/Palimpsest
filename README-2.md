# Palimpsest

Artifact for **"Palimpsest: Erased Data Still Shapes Retrieval-Augmented LLM Systems"**
(anonymous submission, IEEE S&P 2027).

Retrieval-augmented LLM applications store data and also *derive* artifacts from it: graph-index
topology, quantizer codebooks, LLM-written entity descriptions, community summaries, consolidated
memories, and cached answers. Deleting a record's stored copies leaves its influence in these
artifacts. This repository contains

* **Palimpsest**, which finds where a deleted record's influence persists by tracing canary
  records through a pipeline and replaying it with and without the record;
* **three client-side attacks** on that residue: deletion inference, residue recovery, and zombie
  poisoning;
* **LSED** (lineage-scoped exact deletion), a deletion procedure that restores the counterfactual
  state for every state-based interface;
* **experiments on four real systems** (Microsoft GraphRAG, LightRAG, Mem0, and Graphiti), each
  driven through its own deletion API, with and without LSED's principles;
* the **raw results and logs** behind every figure and table in the paper (`paper_results/`).

## Results at a glance

Every run uses Qwen2.5-7B-Instruct (vLLM 0.30.0, greedy decoding) and bge-large-en-v1.5.

| Finding | Result | Paper |
|---|---|---|
| Copy deletion leaves residue in derived artifacts | in every system except a full rebuild, 10 of 10 trials | Fig. 4 |
| Deletion inference from returned scores (IVF-PQ) | AUC up to 0.90; 79% TPR at 1% FPR for one deleted record | Fig. 5a, 5b |
| Residue recovery | 99% of deleted facts from merged descriptions, 98% from memories | Fig. 5c |
| Zombie poisoning | success unchanged by deletion: 60% (graph RAG), 100% (memory) | Fig. 5d |
| LSED | every attack at chance or 0; deletes in 6.4 ms vs. 19.6 s for a rebuild | Table 3 |
| Five LLM backends | every attack persists | Table 4 |
| GraphRAG 3.2.0 | the incremental update keeps deleted documents: 77% of deleted facts recovered, 90% zombie success among the poisoned subjects that retrieval reaches; a re-index removes both from clients, not from the LLM cache or the update's backups | Table 5 |
| LightRAG 1.5.7 | the answer cache returns 97% to 99% of deleted facts under every deletion option, and poison stays effective (100%) | Table 5 |
| Mem0 2.2.1 | the next extraction prompt still contains the deleted conversation: a message that refers back restores 90% of the facts erased by `delete_all` and revives deleted poison for 9 of 10 users | Table 5 |
| Graphiti 0.31.2 | `remove_episode` leaves community summaries, through which clients recover 88% to 89% of deleted facts; a true fact that deleted poison invalidated stays invalid | Table 5, App. H |
| LSED's principles on the real systems | no deleted fact or poison recovered, nothing at rest, Mem0 keeps other facts at the counterfactual rate | Table 5 |
| Quality cost of atomize-then-render (HotpotQA) | +2.4 F1 with retrieved passages (95% CI [-0.4, 5.4]); -7.8 F1 with entity descriptions alone ([-12.7, -3.0]) | App. H |

No counterfactual world in any experiment recovered a deleted fact or accepted the poison.

## Repository layout

```
palimpsest/
  systems/      instrumented systems under test: FAISS, hnswlib, MiniGraphRAG, MiniMemory
  attacks/      deletion inference, residue recovery, zombie poisoning
  defense/      LSED and its history-independent HNSW index
  experiments/  rq1, rq2, rq3, ablations; real_systems (LightRAG, Mem0); graphrag_exp (GraphRAG);
                graphiti_exp (Graphiti); hotpot_quality (LSED's quality cost on HotpotQA)
  real/         one "world" per process for each real system, embedding server, shared helpers
  report/       tables, figure data, LLM-backend summary
  canaries.py   synthetic subjects with high-entropy canary facts
  replay.py     paired deletion and counterfactual worlds
  llm.py        OpenAI-compatible client (vLLM), deterministic and memoized
configs/        smoke (CPU), local (local weights), hf_models, backends (Table 4)
scripts/        GPU runs, vLLM server, backend sweep, real-system drivers, reports,
                deletion timing, bootstrap intervals
paper_figures/  make_final_figures.py renders the paper's figures from results
paper_results/  raw results and logs behind the paper (see paper_results/README.md)
tests/          unit, regression, and real-system plumbing tests (stub LLM; fake_gpu/ stands in for a GPU)
run.py          experiment CLI for the instrumented pipelines
run_real.sh     one command for the real-system experiments: setup, run, report
```

## Quick start: CPU, offline, about 2 minutes

No GPU and no model downloads. Requires Python 3.12.

```bash
pip install -r requirements.txt
make test     # 46 passed, 8 skipped (real-system plumbing tests; see below), about 40 s
make check    # whole pipeline at small scale, then 22 claim checks, about 2 min
```

Or with Docker:

```bash
docker build -t palimpsest . && docker run --rm palimpsest
```

The smoke configuration replaces the LLM with a deterministic offline stand-in and shrinks every
experiment, so its numbers differ from the paper's. `make check` verifies that the paper's
qualitative claims hold:

| Check | Claim |
|---|---|
| C1 | each system leaves residue in exactly the tiers the paper reports |
| C2 | deletion inference beats chance against native deletion |
| C3 | naive deletion leaks canary facts; the counterfactual never does |
| C4 | poison survives naive deletion |
| C5 | Theorem 1: deletion-inference AUC is exactly 0.5 under LSED and under a full rebuild |
| C6 | LSED leaks nothing and removes the poison |
| C7 | more partitions make deletion cheaper and tail latency higher |

The eight skipped tests drive LightRAG, Mem0, GraphRAG, Graphiti, and the HotpotQA experiment end
to end against a stub OpenAI-compatible server (`tests/stub_openai.py`), and two of them run the
whole one-command pipeline with stand-ins for the GPU (`tests/fake_gpu/`). They run once those
libraries are installed (section "Real systems") and then report 54 passed. The stub is not a
language model, so these tests check only model-independent behaviour: every world is valid,
storage-level residue is where the paper says, and counterfactual worlds never contain a canary.

## Regenerating the paper's numbers without a GPU

```bash
python paper_figures/make_final_figures.py --results paper_results/results --outdir figures   # Figures 4 to 6
python -m palimpsest.report.backend_table --results paper_results/results                      # Table 4
make table5-paper      # Table 5 (H100 values): all 19 cells and the paper's claims
make extra-paper       # Graphiti, HotpotQA, deletion latency (Appendix H)
make ci-paper          # 95% intervals for every Table 5 cell
python scripts/report_real.py --run paper_results/a100/table5   # Table 5, A100 values and LSED
```

## Instrumented pipelines on a GPU (Sections 7.2 to 7.7)

### Hardware and software

| | GPU | Stack | Time |
|---|---|---|---|
| Main experiments | 1x NVIDIA B300 (288 GB) | vLLM 0.30.0, torch 2.13.0+cu132, CUDA 13.2 | about 4.5 h |
| Replication | 1x NVIDIA A100 (80 GB) | vLLM 0.10.2, torch 2.8.0, CUDA 12.8 | every qualitative finding reproduced |

```bash
pip install -r requirements-gpu.txt
# or, on a fresh B300: install, verify, download weights, and run the tests
bash scripts/setup_b300.sh
```

`configs/local.yaml` reads local weights from `/workspace/models/`: `Qwen2.5-7B-Instruct`
(generation) and `bge-large-en-v1.5` (embeddings). Edit the paths there, or use
`configs/hf_models.yaml` to load models by Hugging Face id.

### Start the LLM server

```bash
bash scripts/start_vllm.sh
# NVIDIA B300 (sm_103): FlashInfer cannot JIT-compile without nvcc, so the paper's runs used
VLLM_USE_FLASHINFER_SAMPLER=0 bash scripts/start_vllm.sh --attention-backend FLASH_ATTN
```

### Run the experiments

`scripts/final_runs.sh` runs the paper's experiments in four groups; each writes to
`results/<run_name>/`.

| Group | Runs | Paper |
|---|---|---|
| `llm` | RQ1, residue recovery and zombie poisoning, RQ3, ablations, timing, one-sentence rendering | Fig. 4, 5c, 5d, 6; Table 3 |
| `ivf8` | deletion inference, IVF-PQ with eight probed lists | Fig. 5a, 5b |
| `ivf1` | deletion inference, IVF-PQ with one probed list | Fig. 5b |
| `controls` | decoy null and approximate HNSW | Fig. 5b |

```bash
bash scripts/final_runs.sh llm
bash scripts/final_runs.sh ivf8
bash scripts/final_runs.sh ivf1
bash scripts/final_runs.sh controls
python paper_figures/make_final_figures.py --results results --outdir figures
```

Individual stages run through `run.py`; any configuration value can be overridden with `--set`:

```bash
python run.py --config configs/local.yaml rq2 --set graph.render_style=one_sentence --set run_name=test
python run.py -h      # stages and the paper element each produces
```

### LLM-backend sweep (Table 4)

Reruns every LLM-dependent experiment with Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3,
Gemma-2-9B-it, and Qwen2.5-72B-Instruct, and with Qwen2.5-7B-Instruct on the same machine
(`qwen7b_repro`), so that every row of Table 4 comes from one machine. Deletion inference does not
involve the LLM and is not rerun.

```bash
python scripts/run_backends.py --dry-run                 # print the plan
python scripts/check_weights.py                          # every weight shard present and complete
python scripts/run_backends.py                           # all backends, one after another
python -m palimpsest.report.backend_table --results results
python scripts/compare_baseline.py --results results --tol 0.10
```

For each backend the driver starts vLLM, accepts the server only if it serves exactly that model,
runs the experiments, and stops it. It refuses a port that is already in use and resumes where it
stopped. Models and per-backend settings are in `configs/backends.yaml`.

## Real systems (Table 5, Appendix H)

### One command

```bash
cd palimpsest-artifact
tmux new -s pal                       # survives disconnects
bash run_real.sh                      # Table 5: GraphRAG, LightRAG, Mem0; about 5 h with setup
bash run_real.sh --systems graphiti hotpotqa timing   # Graphiti, HotpotQA, latency; about 5 h
```

On a machine with one NVIDIA GPU (at least 40 GB; the paper used an H100 and an A100, both
80 GB), an NVIDIA driver, about 50 GB of disk, and network access, `run_real.sh`

1. checks the GPU and disk, installs `uv`, and builds three environments: vLLM 0.30.0 with
   LightRAG 1.5.7 and Mem0 2.2.1; GraphRAG 3.2.0; and Graphiti 0.31.2 with an embedded FalkorDB
   (GraphRAG and Graphiti pin versions that conflict with the main environment);
2. downloads Qwen2.5-7B-Instruct and bge-large-en-v1.5 (ModelScope, or `HUB=hf`) and verifies
   every weight shard;
3. checks ports, pinned versions, and weights before starting a server;
4. runs the stub-LLM plumbing tests, then each selected experiment; every driver starts and stops
   its own vLLM and runs a smoke test before its full run;
5. writes `results/real_eval/REPORT.md` (Table 5 beside the paper's values, with a verdict for each
   of the paper's claims) and `EXTRA_REPORT.md` (Graphiti, HotpotQA, latency, and 95% intervals for
   Table 5); the exit status is 1 if a model-independent claim fails.

The command is idempotent and resumable: after an interruption, rerun it, and finished setup steps,
stages, and worlds are skipped (`--force` reruns stages; a failed stage is always rerun).

| Command | |
|---|---|
| `bash run_real.sh --profile quick` | same setup, then only the smoke tests (about 15 min per experiment) |
| `bash run_real.sh --systems graphrag` | selected experiments: `lightrag-mem0`, `graphrag`, `graphiti`, `hotpotqa`, `timing` |
| `bash run_real.sh --setup-only` / `--skip-setup` | only the setup / only the experiments |
| `bash run_real.sh --skip-tests --skip-smoke` | no plumbing tests, and no smoke test before the Graphiti and HotpotQA runs |
| `bash run_real.sh --dry-run` | setup and checks, then print the plan; starts no server |
| `bash run_real.sh --lsed only` | add only the LSED conditions to an existing Table 5 run; results go to `real_lsed/` and `graphrag_lsed/`, and the report merges them |
| `bash run_real.sh --lsed skip` | Table 5 without the LSED conditions |

Settings, all optional: `PAL_WORKDIR` (environments and weights; default `/workspace` if writable,
else `~/palimpsest-work`), `PAL_MODELS_DIR`, `PAL_VENV`, `PAL_VENV_GRAPHRAG`, `PAL_VENV_GRAPHITI`,
`HUB=modelscope|hf`, `PAL_OUT` (default `results/real_eval`), `PAL_GPU_UTIL` (default 0.40; 0.60 on
GPUs under 70 GB), `PAL_ATTN_BACKEND` (default `FLASH_ATTN`; `auto` lets vLLM choose),
`PAL_VLLM_PORT`, `CUDA_VISIBLE_DEVICES`. Further options: `python scripts/real_all.py -h`.

Greedy decoding on another GPU or software stack can move individual numbers. The report lists
Table 5 cells that differ from the paper by more than 15 percentage points (`--tol`); the claims,
most of them model-independent, decide the exit status. Canary values and poison are random
12-character tokens, so a model can return one only if some artifact still holds it.

### Systems and deletion procedures

Each system runs through its own API and its own documented deletion. Every *world* (one deletion
or counterfactual scenario) runs in its own process on a copy of a background snapshot; a world
whose documents were not all indexed is listed in `invalid_worlds` and never scored.

| System | Native deletion | Deletion with LSED's principles | Measured |
|---|---|---|---|
| GraphRAG 3.2.0 | no per-document delete: the documented incremental `update` on the remaining corpus, or a full re-index | erase the cache entries written while the removed documents were indexed (their lineage, P1), the update's backups, and the old output tables (P5), then re-index | local-search answers and context; zombie poisoning of 10 indexed subjects; at-rest stores |
| LightRAG 1.5.7 | `adelete_by_doc_id`, with or without `delete_llm_cache=True`, or with extraction caching off | the same, then flush the whole LLM response cache at the deletion epoch (P5) | repeated, reworded, and context-only questions; zombie poisoning of 10 managers; at-rest stores |
| Mem0 2.2.1 | the memories one `add()` returned, or `delete_all` | every `add()` records the message rows it wrote and those its prompt showed the LLM (P1); delete the memories, messages, and history of the deleted call and of every later call that saw it (P5), then replay those calls in order (P3) | memories visible through `get_all` and `search`; zombie poisoning, followed by one more user message that refers back or is unrelated |
| Graphiti 0.31.2 | `remove_episode`, optionally followed by `build_communities` | record the edge validity each episode changes (P1), restore it at deletion, and rebuild the communities (P3) | repeated, reworded, and context-only questions; zombie poisoning of background people with a valid employer edge; edge validity; at-rest stores |

P2 (partitions) and P4 (atomize, then render) would require changing the systems' internals and
are not applied to them. The HotpotQA experiment measures what P4 costs in answer quality instead:
on 300 distractor-dev questions, it compares entity descriptions written per paragraph and merged
by the LLM with descriptions rendered from atomic facts, and then deletes 10% of the paragraphs.

Graphiti 0.31.2 detects communities with synchronous label propagation that never terminates when
two entities are joined only by several edges, which a few poison records create. The Graphiti
worlds cap it at 100 rounds (`palimpsest/real/graphiti_world.py`); where Graphiti's own loop
converges, the result is identical.

### Model-independent claims checked by the reports

| Claims | |
|---|---|
| G1 to G4, L1 to L3, M1 to M4, R0, R1 | Table 5 (GraphRAG, LightRAG, Mem0) |
| S1 to S5 | LSED's principles on GraphRAG, LightRAG, and Mem0 |
| X0 to X5 | Graphiti: valid worlds, community residue, invalidation residue, LSED |
| H1 | HotpotQA: after deletion, LSED holds exactly the counterfactual's entities |

### Outputs under `results/real_eval/`

| File | Content |
|---|---|
| `REPORT.md`, `table5.json` | Table 5 against the paper; claim verdicts |
| `EXTRA_REPORT.md`, `extra.json` | Graphiti, HotpotQA, deletion latency, 95% intervals; claim verdicts |
| `status.json` | state, duration, and command of every stage |
| `real/`, `graphrag/`, `real_lsed/`, `graphrag_lsed/` | Table 5 results: `summary` and per-world `rows` |
| `graphiti/`, `hotpot/`, `timing/` | results of the other experiments |
| `*/worlds/ops/*.out.json` | every world's answers, retrieved contexts, and at-rest scans |
| `logs/` | setup, tests, each driver, vLLM and embedding-server logs |

### Troubleshooting

* A run stopped: rerun the same command. `status.json` shows which stage failed, and
  `logs/<stage>/` holds its logs.
* `port 8000 is already in use`: a driver or a leftover vLLM is running; wait, or
  `pkill -f vllm.entrypoints`.
* `port 8001 is already in use` (embedding server): pass `--emb-port 8011` or any free port.
* vLLM does not start: see `logs/vllm_qwen7b.log`. All drivers use `--max-model-len 32768`, because
  Mem0's extraction prompt is about 6.7k tokens and LightRAG's query budget is 30k.
* Graphiti worlds that fail because the LLM returned malformed JSON to a structured call fail the
  same way on every rerun (greedy decoding); they are listed in `invalid_worlds` and excluded.
* HotpotQA is downloaded from the Hugging Face hub or the official site; if both are blocked, pass
  `--hotpot-data` with a local copy of `hotpot_dev_distractor_v1.json`.

## Results behind the paper

`paper_results/` holds the raw results and logs of every run in the paper; its README maps each
directory to the paper. In short: `results/` (Sections 7.2 to 7.7 and Table 4, B300s), `real2/`,
`graphrag/run2/`, and `zombie_rerun/` (Table 5, H100), `a100/table5/` (Table 5 in parentheses and
the LSED row, A100), and `a100/followup/` (Graphiti, HotpotQA, latency, A100).

## Determinism

Each experiment compares a deletion world with a counterfactual world that never held the record,
so the two must differ only in that record. Every index is built single-threaded, decoding is
greedy, and LLM completions are memoized, which makes both worlds deterministic. Canary facts are
high-entropy random tokens, so the counterfactual world cannot produce them, and any canary seen
after deletion is residue by construction.

A different software stack can change individual greedy completions. Qwen2.5-7B-Instruct on a
second B300 reproduced every deletion outcome and differed only in zombie success against
MiniGraphRAG (50% instead of 60%), as noted under Table 4. GraphRAG's and Graphiti's indexing is
not deterministic across runs (concurrent LLM calls), so each world is compared with itself before
deletion. In the HotpotQA experiment, 0.4% of LSED's re-rendered descriptions differ in wording
from the counterfactual's, consistent with batched decoding in vLLM not being bitwise
deterministic.

## Scope and limitations

* MiniGraphRAG and MiniMemory are instrumented reproductions of graph-RAG and agent-memory
  pipelines with full lineage, not production frameworks; the real-system experiments cover
  GraphRAG, LightRAG, Mem0, and Graphiti. `palimpsest/systems/external_adapters.py` contains thin,
  untested starting points for further systems; no result in the paper uses them.
* Subjects and documents are synthetic, which makes canary facts easier to trace than natural
  attributes.
* The Mem0 zombie result depends on the user's next message: a message that refers back revives
  the poison, an unrelated one does not (0%). Both are reported.
* Qwen2.5-7B rarely judges Graphiti's poison to contradict the true fact: the invalidation residue
  rests on 4 worlds, where the deletion behaviour itself is deterministic code.
* Timing results depend on the host: native-deletion timing AUC was 0.64 on the B300 and 0.86 on
  the A100.

## Responsible use

All experiments use synthetic people and local deployments of open-source software; no
third-party service is contacted. The attacks are released together with LSED, which drives all
three to chance. Disclosure to the maintainers of the affected components is described in the
paper's responsible-disclosure section.

## License

MIT; see `LICENSE`.

## Citation

Citation information will be added after the review process.
