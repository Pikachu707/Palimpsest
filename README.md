# Palimpsest

Artifact for **"Palimpsest: Erased Data Still Shapes Retrieval-Augmented LLM Systems"**
(anonymous submission, IEEE S&P 2027).

Retrieval-augmented LLM applications do not only store data. They *derive* artifacts from it:
graph-index topology, quantizer codebooks, LLM-written entity descriptions, consolidated memories.
Deleting a record's stored copies leaves its influence in these artifacts. This repository contains

* **Palimpsest**, a method that finds where a deleted record's influence persists, by tracing
  canary records through a pipeline and replaying it with and without the record;
* **three client-side attacks** that exploit that residue: deletion inference, residue recovery,
  and zombie poisoning;
* **LSED** (lineage-scoped exact deletion), a defense that makes deletion exact for every
  state-based interface;
* the **raw results and logs** behind every figure and table in the paper.

## Results at a glance

Measured on one NVIDIA B300 with Qwen2.5-7B-Instruct and bge-large-en-v1.5.

| Finding | Result | Paper |
|---|---|---|
| Copy deletion leaves residue in derived artifacts | every system except a full rebuild, in 10/10 trials | Fig. 4 |
| Deletion inference from returned scores (IVF-PQ) | AUC up to 0.90; 79% TPR at 1% FPR for one deleted record | Fig. 5a, 5b |
| Residue recovery | 99% of deleted facts from merged descriptions, 98% from memory | Fig. 5c |
| Zombie poisoning | success unchanged by deletion: 60% (graph RAG), 100% (memory) | Fig. 5d |
| LSED | every attack at chance or 0; deletes in 6.4 ms vs. 19.6 s for a rebuild | Table 3 |
| Five LLM backends | every finding holds | Table 4 |

## Repository layout

```
palimpsest/
  systems/      systems under test: FAISS, hnswlib, MiniGraphRAG, MiniMemory
  attacks/      deletion inference, residue recovery, zombie poisoning
  defense/      LSED and its history-independent HNSW index
  experiments/  one module per research question (rq1, rq2, rq3, ablations)
  report/       tables, figure data, LLM-backend summary
  canaries.py   synthetic subjects with high-entropy canary facts
  replay.py     paired deletion and counterfactual worlds
  llm.py        OpenAI-compatible client (vLLM), deterministic and memoized
configs/        smoke (CPU), local (local weights), hf_models, backends (Table 4)
scripts/        GPU runs, vLLM server, backend sweep, checks
paper_figures/  make_final_figures.py renders the paper's figures from results
paper_results/  raw results and logs behind the paper
tests/          unit and regression tests
run.py          experiment CLI
```

## Quick start: CPU, offline, about 2 minutes

No GPU, no model downloads. Requires Python 3.12.

```bash
pip install -r requirements.txt
make test     # 41 tests, about 25 s
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

## Reproducing the paper on a GPU

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

### Models

`configs/local.yaml` reads local weights from `/workspace/models/`:
`Qwen2.5-7B-Instruct` (generation) and `bge-large-en-v1.5` (embeddings).
Edit the paths there, or use `configs/hf_models.yaml` to load models by Hugging Face id.

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

## LLM-backend sweep (Table 4)

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

## Results behind the paper

`paper_results/` holds the raw results and logs of every run in the paper, in the same layout as a
fresh `results/` directory (see `paper_results/README.md`). Both of these reproduce the paper exactly:

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
zombie success against MiniGraphRAG (50% instead of 60%), as noted under Table 4.

## Scope and limitations

* MiniGraphRAG and MiniMemory are instrumented reproductions of graph-RAG and agent-memory
  pipelines with full lineage, not production frameworks.
  `palimpsest/systems/external_adapters.py` contains thin, untested starting points for real
  systems; no result in the paper uses them.
* Subjects and documents are synthetic, which makes canary facts easier to trace than natural
  attributes.
* Timing results depend on the host: native-deletion timing AUC was 0.64 on the B300 and 0.86 on
  the A100 pilot.

## Responsible use

All experiments use synthetic people and local deployments of open-source software; no third-party
service is contacted. The attacks are released together with LSED, which drives all three to chance.
We are notifying the maintainers of the affected open-source components, as described in the
paper's responsible-disclosure section.

## License

MIT; see `LICENSE`.

## Citation

Citation information will be added after the review process.
