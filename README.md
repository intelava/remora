# remora

Scaffold for ragged batching + W8A16 quantization. Prior accelerator-specific
code has been removed; the TODOs are yours to fill.

## Where to write code
- `remora/engine.py`: build ragged batches, run your model with W8A16, decode outputs.
- `remora/kernels.py`: JaggedTensor helpers and the W8A16 kernels/quantizers to implement.
- `remora/surgery.py`: optional model surgery hooks if you want to swap modules.
- `remora/integration.py`: VLMEvalKit adapter that calls `RemoraEngine.generate_batch`.
- `vibecheck.py`: Friendly facade for `import vibecheck; vibecheck.accelerate(model)`.

## Usage
- Instantiate the engine (after filling in the TODOs):
  ```python
  import vibecheck
  engine = vibecheck.accelerate(model, tokenizer)
  ```
- Wrap for VLMEvalKit:
  ```python
  from remora.integration import VibeCheckModel
  with VibeCheckModel(model, tokenizer, evaluator=engine) as vibe:
      ...
  ```
- Run a quick eval once your kernels are in place: `python run_eval.py --model-path /path/to/model`.

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
- Run VLMEvalKit with batching + prefix caching (after filling in the engine/kernels):
  ```
  python vlmeval_bench.py --preset molmo-7b --datasets MME,TextVQA --batch-size 4
  ```
  Adjust `--datasets` to any VLMEvalKit dataset name (e.g., GQA, MMVet, MathVista).
- Discover presets without editing code: `python vlmeval_bench.py --list-presets`.
- Tune batching from the CLI: `--batch-size`, `--flush-ms`, and `--max-queue` control how aggressively requests are coalesced.

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
