import torch
import torch.nn as nn
from remora.layers import TritonRMSNorm, TritonBinaryHead

def replace_rmsnorm(module):
    count = 0
    for name, child in module.named_children():
        if "RMSNorm" in child.__class__.__name__:
            setattr(module, name, TritonRMSNorm(child))
            count += 1
        else:
            count += replace_rmsnorm(child)
    return count

def optimize_model(model, processor):
    up_id = processor.tokenizer.encode("up")[-1]
    down_id = processor.tokenizer.encode("down")[-1]
    triton_head = TritonBinaryHead(model.lm_head, up_id, down_id)
    model.lm_head = triton_head
    model.vision_tower.vision_model = torch.compile(model.vision_tower.vision_model)
    
    return model
