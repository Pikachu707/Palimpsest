# Palimpsest

Artifact for **"Palimpsest: Erased Data Still Shapes Retrieval-Augmented LLM Systems"**
(anonymous submission).

Retrieval-augmented LLM applications do not only store data. They *derive* artifacts from it:
graph-index topology, quantizer codebooks, LLM-written entity descriptions, consolidated memories,
cached answers. Deleting a record's stored copies leaves its influence in these artifacts.
This repository contains

* **Palimpsest**, a method that finds where a deleted record's influence persists, by tracing
  canary records through a pipeline and replaying it with and without the record;
* **three client-side attacks** that exploit that residue: deletion inference, residue recovery,
  and zombie poisoning;
* **LSED** (lineage-scoped exact deletion), a defense that makes deletion exact for every
  state-based interface;
* **real-system studies** of Microsoft GraphRAG, LightRAG, and Mem0, driven through their own
  deletion procedures;
* the **raw results and logs** behind every figure and table in the paper.

## Results at a glance

| Finding | Result | Paper |
|---|---|---|
| Copy deletion leaves residue in derived artifacts | every system except a full rebuild, in 10/10 trials | Fig. 4 |
| Deletion inference from returned scores (IVF-PQ) | AUC up to 0.90; 79% TPR at 1% FPR for one deleted record | Fig. 5a, 5b |
| Residue recovery | 99% of deleted facts from merged descriptions, 98% from memory | Fig. 5c |
| Zombie poisoning | success unchanged by deletion: 60% (graph RAG), 100% (memory) | Fig. 5d |
| LSED | every attack at chance or 0; deletes in 6.4 ms vs. 19.6 s for a rebuild | Table 3 |
| Five LLM backends | every finding holds | Table 4 |
| Microsoft GraphRAG 3.2.0 | the incremental update silently keeps deleted documents; clients still recover 77% of deleted facts | Table 5 |
| LightRAG 1.5.7 | the answer cache returns 97% to 99% of deleted facts under every deletion option | Table 5 |
| Mem0 2.2.1 | `delete_all` leaves deleted conversations in its history and message tables | Table 5 |

Figures 4 to 6 and Tables 3 and 4 were measured on one NVIDIA B300; Table 5 on one NVIDIA H100.
All runs use Qwen2.5-7B-Instruct and bge-large-en-v1.5 unless stated otherwise.

## Repository layout

```
palimpsest/
  systems/      systems under test: FAISS, hnswlib, MiniGraphRAG, MiniMemory
  attacks/      deletion inference, residue recovery, zombie poisoning
  defense/      LSED and its history-independent HNSW index
  experiments/  one module per research question, plus real_systems.py and graphrag_exp.py
  real/         one process per "world" for LightRAG, Mem0, and GraphRAG; local embedding server
  report/       tables, figure data, LLM-backend summary
  canaries.py   synthetic subjects with high-entropy canary facts
  replay.py     paired deletion and counterfactual worlds
  llm.py        OpenAI-compatible client (vLLM), deterministic and memoized
configs/        smoke (CPU), local (local weights), hf_models, backends (models and vLLM settings)
scripts/        GPU runs, vLLM server, backend sweep, real-system runners, checks
paper_figures/  make_final_figures.py renders the paper's figures from results
paper_results/  raw results and logs behind the paper
tests/          unit, regression, and end-to-end tests (with an offline stub LLM)
run.py          experiment CLI
```

## Quick start: CPU, offline, about 2 minutes

No GPU, no model downloads. Requires Python 3.12.

```bash
pip install -r requirements.txt
make test     # unit, regression, and end-to-end tests, about 1 minute
make check    # whole pipeline at small scale, then 22 claim checks, about 2 minutes
```

Or with Docker: `docker build -t palimpsest . && docker run --rm palimpsest`.

The smoke configuration replaces the LLM with a deterministic offline stand-in and shrinks every
experiment, so its numbers differ from the paper's. `make check` verifies the qualitative claims:

| Check | Claim |
|---|---|
| C1 | each system leaves residue in exactly the tiers the paper reports |
| C2 | deletion inference beats chance against native deletion |
| C3 | naive deletion leaks canary facts; the counterfactual never does |
| C4 | poison survives naive deletion |
| C5 | Theorem 1: deletion-inference AUC is exactly 0.5 under LSED and under a full rebuild |
| C6 | LSED leaks nothing and removes the poison |
| C7 | more partitions make deletion cheaper and tail latency higher |

Tests that need LightRAG, Mem0, or GraphRAG are skipped unless those libraries are installed.

## Reproducing the paper on a GPU

### Software

| | GPU | Stack | Time |
|---|---|---|---|
| Main runs (Figs. 4 to 6, Tables 3, 4) | 1x NVIDIA B300 (288 GB) | vLLM 0.30.0, torch 2.13.0+cu132, CUDA 13.2 | about 4.5 h |
| Real systems (Table 5) | 1x NVIDIA H100 (80 GB) | vLLM 0.30.0, torch 2.13.0 | about 2 h |
| Pilot | 1x NVIDIA A100 (80 GB) | vLLM 0.10.2, torch 2.8.0, CUDA 12.8 | reproduced every qualitative finding |

```bash
pip install -r requirements-gpu.txt
# or, on a fresh B300: install, verify, download weights, and run the tests
bash scripts/setup_b300.sh
```

`configs/local.yaml` reads local weights from `/workspace/models/`: `Qwen2.5-7B-Instruct` and
`bge-large-en-v1.5`. Edit the paths there, or use `configs/hf_models.yaml` to load models by
Hugging Face id.

**GPU-specific setting.** `configs/backends.yaml` sets `attention_backend: FLASH_ATTN`, which the
B300 (sm_103) needs because FlashInfer cannot JIT-compile there without nvcc. On any other GPU
(H100, A100), remove the flag:

```bash
sed -i 's/^attention_backend:.*/attention_backend: null/' configs/backends.yaml
```

### Main experiments (Figures 4 to 6, Table 3)

```bash
bash scripts/start_vllm.sh                      # on a B300, add: --attention-backend FLASH_ATTN
                                                # and set VLLM_USE_FLASHINFER_SAMPLER=0
bash scripts/final_runs.sh llm                  # Fig. 4, 5c, 5d, 6; Table 3
bash scripts/final_runs.sh ivf8                 # Fig. 5a, 5b
bash scripts/final_runs.sh ivf1                 # Fig. 5b
bash scripts/final_runs.sh controls             # Fig. 5b (decoy null, approximate HNSW)
python paper_figures/make_final_figures.py --results results --outdir figures
```

Individual stages run through `run.py`, and any configuration value can be overridden with `--set`:

```bash
python run.py --config configs/local.yaml rq2 --set graph.render_style=one_sentence --set run_name=test
python run.py -h      # stages and the paper element each produces
```

### LLM-backend sweep (Table 4)

Reruns every LLM-dependent experiment with Llama-3.1-8B-Instruct, Mistral-7B-Instruct-v0.3,
Gemma-2-9B-it, and Qwen2.5-72B-Instruct, plus Qwen2.5-7B-Instruct (`qwen7b_repro`), all on one machine.

```bash
python scripts/run_backends.py --dry-run          # print the plan
python scripts/check_weights.py                   # every weight shard present and complete
python scripts/run_backends.py                    # all backends, one after another; resumable
python -m palimpsest.report.backend_table --results results
python scripts/compare_baseline.py --results results --tol 0.10
```

### Real systems (Table 5, Appendix H)

Each system runs through its own deletion procedure with the paper's canary population:

| System | Deletion procedures | Runner | Time (H100) |
|---|---|---|---|
| LightRAG 1.5.7 | `adelete_by_doc_id`; with `delete_llm_cache=True`; with extraction caching off | `run_real_systems.py` | about 75 min, with Mem0 |
| Mem0 2.2.1 | delete the memories one `add()` returned; `delete_all` | `run_real_systems.py` | (included above) |
| Microsoft GraphRAG 3.2.0 | incremental update on the remaining corpus; full re-index | `run_graphrag.py` | about 40 min |

The libraries' dependencies conflict with each other and with vLLM, so each gets its own environment.
vLLM stays in the system Python; both virtualenvs reuse its torch through `--system-site-packages`.

```bash
apt-get install -y build-essential python3-dev python3-venv tmux     # hnswlib compiles from source
pip install --break-system-packages uv
uv pip install --system --break-system-packages vllm sentence-transformers pyyaml pytest

python3 -m venv --system-site-packages /workspace/venv-real
source /workspace/venv-real/bin/activate
uv pip install -r requirements.txt -r requirements-real.txt
deactivate

python3 -m venv --system-site-packages /workspace/venv-graphrag
uv pip install --python /workspace/venv-graphrag/bin/python -r requirements-graphrag.txt
/workspace/venv-graphrag/bin/python -c "import tiktoken; tiktoken.get_encoding('o200k_base')"
```

Run with the experiment environment active:

```bash
source /workspace/venv-real/bin/activate
python scripts/run_real_systems.py --vllm-python /usr/bin/python3 --subjects-per-manager 1
python scripts/run_graphrag.py --vllm-python /usr/bin/python3 \
    --graphrag-python /workspace/venv-graphrag/bin/python --emb-port 8701
```

Both runners start vLLM (and, for GraphRAG, a local OpenAI-compatible embedding server), refuse
ports that are already in use, accept a server only if it serves exactly the requested model, and
run a smoke test before the full experiment. Every world runs in its own process; a world whose
documents were not all indexed, or whose indexing failed, is reported as invalid and never scored.
GraphRAG community reports use a 0.3 frequency penalty and every completion is capped at 4,096
tokens, because without them some reports loop under greedy decoding and never terminate.

### Troubleshooting

| Symptom | Fix |
|---|---|
| `externally-managed-environment` when installing | add `--break-system-packages` (system Python only) |
| hnswlib fails to build (`g++` or `Python.h` missing) | `apt-get install -y build-essential python3-dev` |
| GraphRAG cannot download `o200k_base` | run the `tiktoken.get_encoding` line above once with network access |
| `port ... is already in use` | another service holds the port (often a proxy on 8001); pass `--emb-port` with a free port |
| vLLM fails to start on a non-B300 GPU | set `attention_backend: null` (see above) |
| a prompt exceeds the context length | the runners start vLLM with `--max-model-len 32768`; keep that setting |

## Results behind the paper

`paper_results/` holds the raw results and logs of every run in the paper:

| Directory | Paper |
|---|---|
| `results/final/`, `results/final_onesent/`, `results/di_*` | Figs. 4 to 6, Table 3 |
| `results/be_*/`, `results/backends/` | Table 4 |
| `real/` (run 1), `real2/` (run 2) | Table 5 and Appendix H: LightRAG and Mem0 |
| `graphrag/run2/` (reported), `graphrag/run1/` | Table 5 and Appendix H: GraphRAG |
| `logs/` | run logs, including each vLLM server log |

These commands reproduce the paper's figures and Table 4 exactly from the archived results:

```bash
python paper_figures/make_final_figures.py --results paper_results/results --outdir figures   # Figs. 4 to 6
python -m palimpsest.report.backend_table --results paper_results/results                      # Table 4
```

## Determinism

Each experiment compares a deletion world with a counterfactual world that never held the record.
In our pipelines, LightRAG, and Mem0, indexes are built deterministically, decoding is greedy, and
LLM completions are memoized or cached, so both worlds differ only in the record. Canary facts are
high-entropy random tokens, so the counterfactual world cannot produce them, and any canary seen
after deletion is residue by construction.

Two caveats. Rerunning on another software stack can change individual greedy completions: our
rerun of Qwen2.5-7B-Instruct on a second B300 reproduced every deletion outcome and differed only in
zombie success against MiniGraphRAG (50% instead of 60%). GraphRAG's output varies between runs
(77% to 95% recovery before deletion), so each GraphRAG world is compared before and after its own
deletion.

## Scope and limitations

* MiniGraphRAG and MiniMemory are instrumented reproductions with full lineage; the real-system
  studies cover GraphRAG, LightRAG, and Mem0 at the pinned versions above, whose deletion paths may
  change between releases.
* Subjects and documents are synthetic, which makes canary facts easier to trace than natural attributes.
* Timing results depend on the host: native-deletion timing AUC was 0.64 on the B300 and 0.86 on
  the A100 pilot.
* Zombie poisoning could not be measured on GraphRAG: its local search did not retrieve the poisoned
  entity even before deletion.

## Responsible use

All experiments use synthetic people and local deployments of open-source software; no third-party
service is contacted. The attacks are released together with LSED, which drives all three to chance.
We reported each finding to the maintainers of the affected project, as described in the paper's
responsible-disclosure section.

## License

MIT; see `LICENSE`.

## Citation

Citation information will be added after the review process.
