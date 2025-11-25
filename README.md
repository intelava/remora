# remora

Hyper-optimized evaluation harness for vision-language benchmarks using Triton.

## Usage
- Swap linear layers to Triton and obtain an evaluator:
  ```python
  import vibecheck
  evaluator = vibecheck.accelerate(model, tokenizer)
  ```
- Wrap for VLMEvalKit:
  ```python
  from remora.integration import VibeCheckModel
  with VibeCheckModel(model, tokenizer) as vibe:
      ...
  ```
- Run a quick TPS comparison: `python run_eval.py --model-path /path/to/model`.
- Try an interactive notebook: `notebooks/remora_quickstart.ipynb`.

## What lives where
- `remora/kernels.py`: W8A16 Triton GEMM kernel with autotuning and CPU fallback.
- `remora/surgery.py`: Model hijacker that swaps nn.Linear for TritonBitLinear.
- `remora/engine.py`: Prefix caching and batch-aware generation orchestrator.
- `remora/integration.py`: VLMEvalKit adapter with batching queue to force GPU utilization.
- `vibecheck.py`: Friendly facade for `import vibecheck; vibecheck.accelerate(model)`.

## Preset model shortcuts

All models listed below support vision and can run on 16GB GPU memory:

- `molmo-7b` -> `allenai/Molmo-7B-D-0924` (7B)
- `molmo-7b-o` -> `allenai/Molmo-7B-O-0924` (7B, open weights)
- `smolvlm-base` -> `HuggingFaceTB/SmolVLM-Base` (Base)
- `qwen2.5-vl-7b` -> `Qwen/Qwen2.5-VL-7B-Instruct` (7B)
- `qwen2.5-vl-3b` -> `Qwen/Qwen2.5-VL-3B-Instruct` (3B)
- `qwen2-vl-2b` -> `Qwen/Qwen2-VL-2B-Instruct` (2B)
- `phi-3.5-vision` -> `microsoft/Phi-3.5-vision-instruct` (4.2B)
- `llava-1.5-7b` -> `llava-hf/llava-1.5-7b-hf` (7B)
- `internvl2-4b` -> `OpenGVLab/InternVL2-4B` (4B)
- `idefics2-8b` -> `HuggingFaceM4/idefics2-8b` (8B)

## VLMEvalKit
- Run VLMEvalKit with batching + prefix caching:
  ```
  python vlmeval_bench.py --preset molmo-7b --datasets MME,TextVQA --batch-size 4
  ```
  Adjust `--datasets` to any VLMEvalKit dataset name (e.g., GQA, MMVet, MathVista).
- Discover presets without editing code: `python vlmeval_bench.py --list-presets`.
- Override hardware or avoid surgery: `python vlmeval_bench.py --device cpu --no-surgery`.
- Tune batching from the CLI: `--batch-size`, `--flush-ms`, and `--max-queue` control how aggressively requests are coalesced.
- Narrow surgery targets if needed: `--surgery-include decoder.layers.0,attention`.

## Install as a package
- Editable dev install:
  ```
  pip install -e .
  ```
- With VLMEvalKit extra:
  ```
  pip install -e ".[eval]"
  ```
- Then import anywhere:
  ```python
  import vibecheck
  evaluator = vibecheck.accelerate(model, tokenizer)
  ```
