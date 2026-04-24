"""Demo: LLaVA-OneVision playing Pong with Remora optimizations.

Runs for 100 frames and prints per-frame latency against a 30 FPS target.

Usage:
    python demos/demo_atari.py
"""

from __future__ import annotations

import numpy as np
import torch
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration

from remora.optimizer import optimize_model

MODEL_ID = "llava-hf/llava-onevision-qwen2-0.5b-si-hf"
TARGET_MS = 1000.0 / 30.0


def _make_prompt(processor) -> str:
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": (
                    "You are a Pong agent. The ball is moving towards you. "
                    "Should you move the paddle up or down? Reply with one word."
                )},
                {"type": "image"},
            ],
        }
    ]
    return processor.apply_chat_template(conversation, add_generation_prompt=True)


def run() -> None:
    print("=" * 60)
    print("  Remora — Atari Pong Demo")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device : {device}")
    print(f"Model  : {MODEL_ID}\n")

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.float16, device_map=device,
    ).eval()
    model = optimize_model(model, processor)

    prompt = _make_prompt(processor)
    up_id = processor.tokenizer.encode("up")[-1]
    down_id = processor.tokenizer.encode("down")[-1]

    # ---- forward hook to capture last hidden state
    container: dict = {}
    hook = model.model.language_model.norm.register_forward_hook(
        lambda m, i, o: container.__setitem__("payload", o)
    )

    try:
        import gymnasium as gym
        env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
        obs, _ = env.reset()
    except Exception as exc:
        print(f"Warning: could not load Atari env ({exc}). Using synthetic frames.")
        env = None
        obs = np.zeros((210, 160, 3), dtype=np.uint8)

    print(f"\n{'Frame':>5}  {'Latency':>10}  {'Action':>6}  {'Status':>6}")
    print("-" * 40)

    latencies: list[float] = []
    n_frames = 100

    for i in range(n_frames):
        img = torch.tensor(obs, dtype=torch.float16, device=device) / 255.0
        inputs = processor(images=img, text=prompt, return_tensors="pt").to(device, torch.float16)

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        with torch.no_grad():
            model(**inputs)
            token_id = int(model.lm_head(container["payload"])[0, 0].item())
        end.record()
        torch.cuda.synchronize()

        latency = start.elapsed_time(end)
        if i >= 5:
            latencies.append(latency)

        env_action = 2 if token_id == up_id else (3 if token_id == down_id else 0)
        action_str = "UP" if env_action == 2 else "DOWN" if env_action == 3 else "NOOP"
        status = "PASS" if latency <= TARGET_MS else "FAIL"

        print(f"{i:>5}  {latency:>8.2f}ms  {action_str:>6}  {status:>6}")

        if env is not None:
            obs, _, term, trunc, _ = env.step(env_action)
            if term or trunc:
                obs, _ = env.reset()

    hook.remove()
    if env is not None:
        env.close()

    if latencies:
        avg = sum(latencies) / len(latencies)
        print("-" * 40)
        print(f"Average latency : {avg:.2f} ms  ({1000/avg:.1f} FPS)")
        print(f"30 FPS target   : {'✓ ACHIEVED' if avg <= TARGET_MS else '✗ NOT MET'}")


if __name__ == "__main__":
    run()
