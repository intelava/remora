import torch
import torch.nn as nn
from typing import List
import cv2
import numpy as np

from .layers import W8A16Linear

class VisionVLM(nn.Module):
    """
    A mock Vision-Language Model that now includes a simple vision "tower".
    It processes flattened image tensors directly, replacing the old text
    embedding layer. This is more representative of a VLM's vision pipeline.
    """
    def __init__(self, image_size, hidden_dim, num_layers, num_actions):
        super().__init__()
        
        flat_image_features = image_size[0] * image_size[1]
        
        torch.manual_seed(42) # for reproducibility
        
        # A simple "vision tower": a linear projection from flattened image to hidden_dim
        # In a real VLM, this would be a complex series of ConvNets (e.g., a ViT).
        # We use our custom W8A16Linear to ensure the entire model uses our kernel.
        vision_proj_layer = nn.Linear(flat_image_features, hidden_dim, bias=False)
        self.vision_projection = W8A16Linear(flat_image_features, hidden_dim, bias=False, linear_layer=vision_proj_layer)
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            cpu_linear = nn.Linear(hidden_dim, hidden_dim, bias=False)
            self.layers.append(W8A16Linear(hidden_dim, hidden_dim, bias=False, linear_layer=cpu_linear))
            
        # The final "head" of the model that predicts an action.
        cpu_head = nn.Linear(hidden_dim, num_actions, bias=False)
        self.head = W8A16Linear(hidden_dim, num_actions, bias=False, linear_layer=cpu_head)

    def forward(self, image_flat: torch.Tensor):
        """
        Forward pass for a preprocessed, flattened image tensor.
        Args:
            image_flat (torch.Tensor): A 2D tensor of shape (batch_size, height * width).
        """
        # (batch, H*W) -> (batch, hidden_dim)
        x = self.vision_projection(image_flat)
        x = nn.functional.relu(x)

        for layer in self.layers:
            x = layer(x)
            x = nn.functional.relu(x)
            
        # (batch, hidden_dim) -> (batch, num_actions)
        logits = self.head(x)
        return logits

class RemoraEngine:
    """
    Refactored engine to process image frames for the Atari demo.
    It now includes preprocessing and a `generate` method to simulate a VLM's response.
    """
    def __init__(self, image_size=(84, 84), hidden_dim=2048, num_layers=6, num_actions=2):
        print("Initializing RemoraEngine with a Vision VLM...")
        self.image_size = image_size
        self.action_map = {0: "UP", 1: "DOWN"} # Map model output index to a string
        
        self.model = VisionVLM(image_size, hidden_dim, num_layers, num_actions).cuda()
        self.model.eval()
        self._warmup()
        print("Engine ready.")
    
    def _preprocess(self, frame: np.ndarray) -> torch.Tensor:
        """
        Preprocesses a raw game frame for the model.
        Args:
            frame (np.ndarray): The game frame (210, 160, 3).
        Returns:
            torch.Tensor: A preprocessed tensor (1, H, W).
        """
        # Convert to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to the model's expected input size
        resized_frame = cv2.resize(gray_frame, self.image_size, interpolation=cv2.INTER_AREA)
        # Convert to torch tensor and add batch dimension
        tensor_frame = torch.from_numpy(resized_frame).unsqueeze(0).float()
        return tensor_frame

    def _warmup(self):
        """Performs a dummy run to compile Triton kernels."""
        print("Warming up Triton kernels...")
        try:
            dummy_frame = np.zeros((210, 160, 3), dtype=np.uint8)
            self.generate(dummy_frame, prompt="Warmup")
        except Exception as e:
            print(f"Warning: Warm-up failed. This might happen on CPU-only environments. Error: {e}")
        torch.cuda.synchronize()

    @torch.inference_mode()
    def generate(self, frame: np.ndarray, prompt: str):
        """
        Simulates a full VLM generation loop for a single frame.
        The prompt is ignored but included for API consistency.

        Args:
            frame (np.ndarray): The raw game frame from the environment.
            prompt (str): A text prompt (e.g., "Track the ball").

        Returns:
            str: The predicted action as a string ("UP" or "DOWN").
        """
        # 1. Preprocess the image frame
        preprocessed_frame = self._preprocess(frame).cuda()
        
        # 2. Flatten the image for the vision projection layer
        flat_frame = preprocessed_frame.flatten(1)

        # 3. Run the model
        logits = self.model(flat_frame)
        
        # 4. "Generate" text: Convert the highest logit index to an action string
        action_idx = torch.argmax(logits, dim=-1).item()
        
        return self.action_map.get(action_idx, "DOWN") # Default to DOWN