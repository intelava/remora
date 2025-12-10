import torch
import torch.nn as nn
from typing import List
import cv2
import numpy as np

from .layers import W8A16Linear

class VisionVLM(nn.Module):
    def __init__(self, image_size, hidden_dim, num_layers, num_actions):
        super().__init__()
        
        flat_image_features = image_size[0] * image_size[1]
        
        torch.manual_seed(42)
        
        vision_proj_layer = nn.Linear(flat_image_features, hidden_dim, bias=False)
        self.vision_projection = W8A16Linear(flat_image_features, hidden_dim, bias=False, linear_layer=vision_proj_layer)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            cpu_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.layers.append(W8A16Linear(hidden_dim, hidden_dim, bias=False, linear_layer=cpu_linear))
            
        cpu_head = nn.Linear(hidden_dim, num_actions, bias=False)
        self.head = W8A16Linear(hidden_dim, num_actions, bias=False, linear_layer=cpu_head)

    def forward(self, image_flat: torch.Tensor):
        x = self.vision_projection(image_flat)
        x = nn.functional.relu(x)

        for layer in self.layers:
            x = layer(x)
            x = nn.functional.relu(x)
            
        logits = self.head(x)
        return logits

class RemoraEngine:
    def __init__(self, image_size=(84, 84), hidden_dim=2048, num_layers=6, num_actions=2):
        print("Initializing RemoraEngine with a Vision VLM...")
        self.image_size = image_size
        self.action_map = {0: "UP", 1: "DOWN"}
        
        self.model = VisionVLM(image_size, hidden_dim, num_layers, num_actions).cuda()
        self.model.eval()
        self._warmup()
        print("Engine ready.")
    
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        resized_frame = cv2.resize(gray_frame, self.image_size, interpolation=cv2.INTER_AREA)
        tensor_frame = torch.from_numpy(resized_frame).unsqueeze(0).float()
        return tensor_frame

    def _warmup(self):
        print("Warming up Triton kernels...")
        try:
            dummy_frame = np.zeros((210, 160, 3), dtype=np.uint8)
            self.generate(dummy_frame, prompt="Warmup")
        except Exception as e:
            print(f"Warning: Warm-up failed. This might happen on CPU-only environments. Error: {e}")
        torch.cuda.synchronize()

    @torch.inference_mode()
    def generate(self, frame: np.ndarray, prompt: str):
        preprocessed_frame = self._preprocess(frame).cuda()
        
        flat_frame = preprocessed_frame.flatten(1)

        logits = self.model(flat_frame)
        
        action_idx = torch.argmax(logits, dim=-1).item()
        
        return self.action_map.get(action_idx, "DOWN")