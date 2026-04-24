from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .layers import W8A16Linear
from .ragged import JaggedTokenIds, pack_token_ids, pad_jagged_token_ids


# ---------------------------------------------------------------------------
# Text generation engine
# ---------------------------------------------------------------------------

@dataclass
class GenerationRequest:
    prompt: str
    max_new_tokens: int = 32


class RemoraEngine:
    """Batched text generation with ragged sequence packing.

    Accepts a plain HuggingFace-style model + tokenizer so it can wrap any
    causal LM without modification.
    """

    def __init__(self, model: nn.Module, tokenizer) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self._pad_id: int = int(getattr(tokenizer, "pad_token_id", None) or 0)

    def build_ragged_batch(self, requests: List[GenerationRequest]) -> Dict[str, JaggedTokenIds]:
        sequences = [
            torch.tensor(self.tokenizer.encode(r.prompt), dtype=torch.long)
            for r in requests
        ]
        return {"input_ids_jagged": pack_token_ids(sequences)}

    def generate_batch(self, requests: List[GenerationRequest]) -> Dict[int, Dict]:
        batch = self.build_ragged_batch(requests)
        jagged = batch["input_ids_jagged"]
        input_ids, attn_mask = pad_jagged_token_ids(jagged, pad_token_id=self._pad_id)

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=input_ids,
                attention_mask=attn_mask,
            )

        return {
            i: {
                "tokens": output_ids[i],
                "text": self.tokenizer.decode(output_ids[i], skip_special_tokens=True),
            }
            for i in range(len(requests))
        }


# ---------------------------------------------------------------------------
# Atari vision engine (lightweight custom model)
# ---------------------------------------------------------------------------

class VisionVLM(nn.Module):
    """Lightweight W8A16-quantized MLP that maps flattened grayscale frames to action logits."""

    def __init__(
        self,
        image_size: tuple[int, int] = (84, 84),
        hidden_dim: int = 2048,
        num_layers: int = 6,
        num_actions: int = 2,
    ) -> None:
        super().__init__()
        flat_size = image_size[0] * image_size[1]
        torch.manual_seed(42)

        self.vision_proj = W8A16Linear(flat_size, hidden_dim, linear_layer=nn.Linear(flat_size, hidden_dim, bias=False))

        self.layers = nn.ModuleList(
            W8A16Linear(hidden_dim, hidden_dim, linear_layer=nn.Linear(hidden_dim, hidden_dim, bias=False))
            for _ in range(num_layers)
        )

        self.head = W8A16Linear(hidden_dim, num_actions, linear_layer=nn.Linear(hidden_dim, num_actions, bias=False))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.vision_proj(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.head(x)


class AtariVisionEngine:
    """Inference engine for Atari frame-to-action using the lightweight VisionVLM."""

    def __init__(
        self,
        image_size: tuple[int, int] = (84, 84),
        hidden_dim: int = 2048,
        num_layers: int = 6,
        num_actions: int = 2,
    ) -> None:
        self.image_size = image_size
        self.action_map = {0: "UP", 1: "DOWN"}
        self.model = VisionVLM(image_size, hidden_dim, num_layers, num_actions).cuda()
        self.model.eval()
        self._warmup()

    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized = cv2.resize(gray, self.image_size, interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized).unsqueeze(0).float()

    def _warmup(self) -> None:
        try:
            dummy = np.zeros((210, 160, 3), dtype=np.uint8)
            self.generate(dummy)
            torch.cuda.synchronize()
        except Exception as exc:
            print(f"Warmup warning (CPU-only?): {exc}")

    @torch.inference_mode()
    def generate(self, frame: np.ndarray) -> str:
        t = self._preprocess(frame).cuda()
        logits = self.model(t.flatten(1))
        return self.action_map.get(int(torch.argmax(logits, dim=-1).item()), "DOWN")
