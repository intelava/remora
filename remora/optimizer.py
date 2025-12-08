import torch.nn as nn
from .layers import W8A16Linear

def optimize_model(model: nn.Module, layers_to_skip=None):
    """
    Recursively iterates through a model and replaces specified nn.Linear layers
    with Remora's W8A16Linear layer for GPU acceleration.

    Args:
        model (nn.Module): The PyTorch model to optimize.
        layers_to_skip (list, optional): A list of layer names (or substrings)
                                         to skip during optimization. For example,
                                         ['lm_head'] might be skipped if you want
                                         to preserve its original precision.
    """
    if layers_to_skip is None:
        layers_to_skip = []
        
    for name, module in model.named_children():
        # Check if the current layer name contains any substring from the skip list
        if any(skip_name in name for skip_name in layers_to_skip):
            print(f"--> Skipping optimization for module: {name}")
            continue

        # If we find a Linear layer, replace it
        if isinstance(module, nn.Linear):
            print(f"--> Optimizing layer: {name}")
            # Create a W8A16Linear layer from the existing nn.Linear layer
            remora_layer = W8A16Linear(
                in_features=module.in_features,
                out_features=module.out_features,
                bias=module.bias is not None,
                linear_layer=module
            )
            # Replace the original layer with the new optimized one
            setattr(model, name, remora_layer)
        
        # Recurse into submodules
        elif len(list(module.children())) > 0:
            print(f"Entering module: {name}")
            optimize_model(module, layers_to_skip)

    return model
