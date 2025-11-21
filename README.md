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
  vibe = VibeCheckModel(model, tokenizer)
  ```
- Run a quick TPS comparison: `python run_eval.py --model-path /path/to/model`.

## What lives where
- `remora/kernels.py`: W8A16 Triton GEMM kernel with autotuning and CPU fallback.
- `remora/surgery.py`: Model hijacker that swaps nn.Linear for TritonBitLinear.
- `remora/engine.py`: Prefix caching and batch-aware generation orchestrator.
- `remora/integration.py`: VLMEvalKit adapter with batching queue to force GPU utilization.
- `vibecheck.py`: Friendly facade for `import vibecheck; vibecheck.accelerate(model)`.

## Preset model shortcuts
- `molmo-7b` -> `allenai/Molmo-7B-D-0924`
- `qwen2.5-vl-7b` -> `Qwen/Qwen2.5-VL-7B-Instruct`

## VLMEvalKit
- Run VLMEvalKit with batching + prefix caching:
  ```
  python vlmeval_bench.py --preset molmo-7b --datasets MME,TextVQA --batch-size 4
  ```
  Adjust `--datasets` to any VLMEvalKit dataset name (e.g., GQA, MMVet, MathVista).
