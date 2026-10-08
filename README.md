# Palimpsest

Artifact for **"Palimpsest: Erased Data Still Shapes Retrieval-Augmented LLM Systems"**

Retrieval-augmented LLM applications do not only store data. They *derive* artifacts from it:
graph-index topology, quantizer codebooks, LLM-written entity descriptions, consolidated memories,
cached answers. Deleting a record's stored copies leaves its influence in these artifacts. This
repository contains

* **Palimpsest**, a method that finds where a deleted record's influence persists, by tracing
  canary records through a pipeline and replaying it with and without the record;
* **three client-side attacks** that exploit that residue: deletion inference, residue recovery,
  and zombie poisoning;
* **LSED** (lineage-scoped exact deletion), a defense that makes deletion exact for every
  state-based interface;
* **real-system experiments** that drive Microsoft GraphRAG, LightRAG, and Mem0 through their own
  deletion APIs;
* the **raw results and logs** behind every figure and table in the paper.

## Results at a glance

Instrumented pipelines: one NVIDIA B300. Real systems: one NVIDIA H100. Every run uses
Qwen2.5-7B-Instruct (vLLM 0.30.0, greedy decoding) and bge-large-en-v1.5.

| Finding | Result | Paper |
|---|---|---|
| Copy deletion leaves residue in derived artifacts | every system except a full rebuild, in 10/10 trials | Fig. 4 |
| Deletion inference from returned scores (IVF-PQ) | AUC up to 0.90; 79% TPR at 1% FPR for one deleted record | Fig. 5a, 5b |
| Residue recovery | 99% of deleted facts from merged descriptions, 98% from memory | Fig. 5c |
| Zombie poisoning | success unchanged by deletion: 60% (graph RAG), 100% (memory) | Fig. 5d |
| LSED | every attack at chance or 0; deletes in 6.4 ms vs. 19.6 s for a rebuild | Table 3 |
| Five LLM backends | every finding holds | Table 4 |
| GraphRAG 3.2.0 | incremental update keeps deleted documents: 77% of deleted facts recovered, poison still effective (90%); a re-index removes both from clients, not from the LLM cache or update backups | Table 5, App. H |
| LightRAG 1.5.7 | the answer cache returns 97% to 99% of deleted facts under every deletion option and keeps poison effective (100%) | Table 5, App. H |
| Mem0 2.2.1 | deletion hides memories from clients, but the next extraction prompt still contains the deleted conversation: a message that refers back restores 90% of the facts erased by `delete_all` and revives deleted poison for 9 of 10 users | Table 5, App. H |

No counterfactual world in any experiment recovered a deleted fact or accepted the poison.

## Repository layout

```
palimpsest/
  systems/      instrumented systems under test: FAISS, hnswlib, MiniGraphRAG, MiniMemory
  attacks/      deletion inference, residue recovery, zombie poisoning
  defense/      LSED and its history-independent HNSW index
  experiments/  rq1, rq2, rq3, ablations; real_systems (LightRAG, Mem0); graphrag_exp (GraphRAG)
  real/         one "world" per process for each real system, embedding server, shared helpers
  report/       tables, figure data, LLM-backend summary
  canaries.py   synthetic subjects with high-entropy canary facts
  replay.py     paired deletion and counterfactual worlds
  llm.py        OpenAI-compatible client (vLLM), deterministic and memoized
configs/        smoke (CPU), local (local weights), hf_models, backends (Table 4)
scripts/        GPU runs, vLLM server, backend sweep, real-system drivers, checks
paper_figures/  make_final_figures.py renders the paper's figures from results
paper_results/  raw results and logs behind the paper
tests/          unit, regression, and real-system plumbing tests (stub LLM)
run.py          experiment CLI for the instrumented pipelines
```

## Quick start: CPU, offline, about 2 minutes

No GPU, no model downloads. Requires Python 3.12.

```bash
pip install -r requirements.txt
make test     # 43 passed, 3 skipped (real-system plumbing tests; see below), about 40 s
make check    # whole pipeline at small scale, then 22 claim checks, about 2 min
```

Or with Docker:

```bash
docker build -t palimpsest . && docker run --rm palimpsest
```

The smoke configuration replaces the LLM with a deterministic offline stand-in and shrinks every
experiment, so its numbers differ from the paper's. `make check` verifies that the paper's
qualitative claims still hold:

| Check | Claim |
|---|---|
| C1 | each system leaves residue in exactly the tiers the paper reports |
| C2 | deletion inference beats chance against native deletion |
| C3 | naive deletion leaks canary facts; the counterfactual never does |
| C4 | poison survives naive deletion |
| C5 | Theorem 1: deletion-inference AUC is exactly 0.5 under LSED and under a full rebuild |
| C6 | LSED leaks nothing and removes the poison |
| C7 | more partitions make deletion cheaper and tail latency higher |

The three skipped tests drive LightRAG, Mem0, and GraphRAG end to end against a stub
OpenAI-compatible server (`tests/stub_openai.py`). They run once those libraries are installed
(section "Real systems", step 5) and then report 46 passed. The stub is not a language model, so
they check only model-independent behaviour: every world is valid, storage-level residue is
where the paper says, deletion hides poisoned Mem0 memories while the deleted conversation stays
in the next extraction prompt, and counterfactual worlds never contain a canary.

## Instrumented pipelines on a GPU (Sections 7.2 to 7.7)

### Hardware and software

| | GPU | Stack | Time |
|---|---|---|---|
| Final runs | 1x NVIDIA B300 (288 GB) | vLLM 0.30.0, torch 2.13.0+cu132, CUDA 13.2 | about 4.5 h |
| Pilot | 1x NVIDIA A100 (80 GB) | vLLM 0.10.2, torch 2.8.0, CUDA 12.8 | reproduced every qualitative finding |

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

`scripts/final_runs.sh` runs the paper's experiments in four groups; each writes to `results/<run_name>/`.

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
Gemma-2-9B-it, and Qwen2.5-72B-Instruct, plus a rerun of Qwen2.5-7B-Instruct on the same machine
(`qwen7b_repro`), so that every row of Table 4 comes from one machine. Deletion inference does not
involve the LLM and is not rerun.

```bash
python scripts/run_backends.py --dry-run                 # print the plan
python scripts/check_weights.py                          # every weight shard present and complete
python scripts/run_backends.py                           # all backends, one after another
python -m palimpsest.report.backend_table --results results
python scripts/compare_baseline.py --results results --tol 0.10
```

For each backend the driver starts vLLM, accepts the server only if it serves exactly that
model, runs the experiments, and stops it. It refuses a port that is already in use and resumes
where it stopped: finished backends are skipped. Models and per-backend settings are listed in
`configs/backends.yaml`.

## Real systems on one H100 (Table 5, Appendix H)

Each real system is driven through its own API and its own documented deletion procedure, with the
paper's canary population, so any canary seen after deletion is residue by construction. Every
*world* (one deletion or counterfactual scenario) runs in its own process on a copy of a background
snapshot; a world whose documents were not all indexed is reported in `invalid_worlds` and never
scored. The drivers start vLLM themselves, run a two-subject smoke test that must pass, run the
full experiment, and stop vLLM.

### Setup (fresh H100 server, about 30 minutes)

```bash
cd /workspace && unzip -q palimpsest-artifact.zip && cd palimpsest-artifact
mkdir -p logs                                         # the drivers write their logs here
tmux new -s pal                                       # runs take up to an hour; survive disconnects

# 1. main environment: vLLM, the instrumented pipelines, LightRAG and Mem0
pip install -U uv
uv venv -p 3.12 /workspace/venv && source /workspace/venv/bin/activate
uv pip install "vllm==0.30.0" --torch-backend=auto
uv pip install -r requirements.txt -r requirements-real.txt openai sentence-transformers modelscope pyyaml

# 2. GraphRAG in its own environment (its pins conflict with requirements.txt)
uv venv -p 3.12 /workspace/venv-graphrag
VIRTUAL_ENV=/workspace/venv-graphrag uv pip install -r requirements-graphrag.txt sentence-transformers pyyaml

# 3. weights, at the paths in configs/local.yaml (or huggingface-cli download ... --local-dir ...)
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir /workspace/models/Qwen2.5-7B-Instruct
modelscope download --model BAAI/bge-large-en-v1.5   --local_dir /workspace/models/bge-large-en-v1.5

# 4. stub-LLM plumbing tests, no GPU (about 3 minutes)
PAL_GRAPHRAG_PYTHON=/workspace/venv-graphrag/bin/python python -m pytest tests/ -q   # 46 passed
```

### Run

Run the drivers **one after another**: each starts its own vLLM on port 8000.

```bash
# LightRAG and Mem0: residue recovery, zombie poisoning, at-rest scans (about 75 minutes).
# --subjects-per-manager 1 gives every subject its own manager: 10 LightRAG zombie targets (run 2)
python scripts/run_real_systems.py --subjects-per-manager 1 --out results/real 2>&1 | tee logs/real_run.log

# Microsoft GraphRAG: recovery and zombie poisoning (about 60 minutes)
python scripts/run_graphrag.py --out results/graphrag \
    --vllm-python /workspace/venv/bin/python \
    --graphrag-python /workspace/venv-graphrag/bin/python 2>&1 | tee logs/graphrag_run.log
```

Subsets: `run_real_systems.py --system {lightrag,mem0,mem0-zombie}` and
`run_graphrag.py --only {recovery,zombie}`. The zombie reruns of Table 5 used

```bash
python scripts/run_real_systems.py --system mem0-zombie --out results/real_zombie        # 3 minutes
python scripts/run_graphrag.py --only zombie --out results/graphrag_zombie \
    --vllm-python /workspace/venv/bin/python --graphrag-python /workspace/venv-graphrag/bin/python   # 22 minutes
```

### What each experiment does

| System | Deletion procedures | Measured |
|---|---|---|
| LightRAG 1.5.7 | `adelete_by_doc_id` (`default`), with `delete_llm_cache=True` (`thorough`), and with extraction caching off (`nocache`) | repeated, reworded, and context-only questions; zombie poisoning of 10 managers; at-rest stores |
| Mem0 2.2.1 | the memories the deleted conversation's `add()` returned (`lineage`), or `delete_all` for the user (`user`) | memories visible through `get_all` and `search`; history, message, and vector tables at rest |
| Mem0 2.2.1, zombie | as above, for a poisoned third conversation among five; then the user sends one more message, referring back (`ref`) or unrelated (`neutral`) | poison success before deletion, right after it, and after that message; whether the deleted conversation is in the next extraction prompt |
| GraphRAG 3.2.0 | no per-document delete: the documented incremental `update` on the remaining corpus, or a full re-index (reusing the LLM cache) | local-search answers and retrieved context; zombie poisoning of 10 indexed subjects, with retrieval diagnostics; at-rest stores |

GraphRAG uses a local OpenAI-compatible embedding server (`palimpsest/real/embed_server.py`,
bge-large-en-v1.5, port 8001; `--emb-port` changes it).

### Outputs

| File | Content |
|---|---|
| `results/real/{lightrag,mem0,mem0_zombie,meta}.json` | `summary` (the numbers of Table 5) and per-world `rows` |
| `results/graphrag/{graphrag,meta}.json` | `summary.recovery`, `summary.zombie`, `summary.counterfactual` |
| `results/*/.../ops/*.out.json` | every world's answers, retrieved contexts, and at-rest scans |
| `logs/vllm_qwen7b.log`, `logs/real_*.log`, `logs/graphrag_*.log`, `logs/embed_server.log` | server and driver logs |

Print the summaries:

```bash
python -c "import json; print(json.dumps(json.load(open('results/real/mem0_zombie.json'))['summary'], indent=1))"
python -c "import json; print(json.dumps(json.load(open('results/graphrag/graphrag.json'))['summary'], indent=1))"
```

Zombie summaries report `before`, `after`, `n_effective` (targets the poison reached before
deletion), and `after_given_effective`. GraphRAG additionally reports `target_in_context_*` and
`poison_in_context_*`; Mem0 reports `deleted` (right after deletion) and `poison_in_next_prompt`.

### Troubleshooting

* `port 8000 is already in use`: another driver or a leftover vLLM is running. Wait for it, or
  `pkill -f vllm.entrypoints`, then rerun.
* `port 8001 is already in use`: pass `--emb-port 8011` (or any free port) to `run_graphrag.py`.
* vLLM does not start: see `logs/vllm_qwen7b.log`. Both drivers use `--max-model-len 32768`,
  because Mem0's extraction prompt is about 6.7k tokens and LightRAG's query budget is 30k.
* An interrupted GraphRAG run resumes: rerun the same command, and completed worlds are reused.

## Results behind the paper

`paper_results/` holds the raw results and logs of every run in the paper (see
`paper_results/README.md`):

| Directory | Paper |
|---|---|
| `results/final/`, `results/final_onesent/`, `results/di_*` | Sections 7.2 to 7.7, Figures 4 to 6, Table 3 (one B300) |
| `results/be_<tag>/`, `results/be_<tag>_onesent/` | Table 4 (backend sweep, a second B300) |
| `real/`, `real2/` | Table 5, Appendix H: LightRAG and Mem0, runs 1 and 2 (one H100) |
| `graphrag/` | Table 5, Appendix H: GraphRAG recovery (run 2 reported; run 1 earlier settings) |
| `zombie_rerun/` | Table 5, Appendix H: GraphRAG and Mem0 zombie poisoning (one H100) |
| `logs/` | run logs, including every vLLM server log |

These reproduce the paper's figures and Table 4 exactly:

```bash
python paper_figures/make_final_figures.py --results paper_results/results --outdir figures   # Figs. 4 to 6
python -m palimpsest.report.backend_table --results paper_results/results                      # Table 4
```

## Determinism

Each experiment compares a deletion world with a counterfactual world that never held the record,
so the two must differ only in that record. Every index is built single-threaded, decoding is
greedy, and LLM completions are memoized, which makes both worlds deterministic. Canary facts are
high-entropy random tokens, so the counterfactual world cannot produce them and any canary seen
after deletion is residue by construction.

Rerunning on a different software stack can change individual greedy completions. Our rerun of
Qwen2.5-7B-Instruct on a second B300 reproduced every deletion outcome exactly and differed only in
zombie success against MiniGraphRAG (50% instead of 60%), as noted under Table 4. GraphRAG's
indexing is not deterministic across runs (concurrent LLM calls), so each GraphRAG world is
compared before and after its own deletion.

## Scope and limitations

* MiniGraphRAG and MiniMemory are instrumented reproductions of graph-RAG and agent-memory
  pipelines with full lineage, not production frameworks; the real-system experiments cover
  GraphRAG, LightRAG, and Mem0. `palimpsest/systems/external_adapters.py` contains thin, untested
  starting points for further systems; no result in the paper uses them.
* Subjects and documents are synthetic, which makes canary facts easier to trace than natural
  attributes.
* The Mem0 zombie result depends on the user's next message: a message that refers back revives
  the poison, an unrelated one does not (0%). Both are reported.
* Timing results depend on the host: native-deletion timing AUC was 0.64 on the B300 and 0.86 on
  the A100 pilot.

## Responsible use

All experiments use synthetic people and local deployments of open-source software; no third-party
service is contacted. The attacks are released
together with LSED, which drives all three to chance. We are notifying the maintainers of the
affected open-source components, as described in the paper's responsible-disclosure section.

## License

MIT; see `LICENSE`.

## Citation

Citation information will be added after the review process.
