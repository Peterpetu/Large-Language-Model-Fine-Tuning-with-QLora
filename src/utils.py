import torch

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {trainable_params} || All params: {total_params} || Trainable%: {100 * trainable_params / total_params}")

def print_weight_range(model):
    """
    Prints the minimum and maximum values of the model weights.
    """
    min_val = min(p.data.min().item() for p in model.parameters() if p.requires_grad)
    max_val = max(p.data.max().item() for p in model.parameters() if p.requires_grad)
    print(f"Weight range: {min_val} to {max_val}")

def print_quantized_weights(model):
    """
    Prints the weights of quantized layers in the model.
    """
    for name, param in model.named_parameters():
        if 'quantize' in name:
            print(f"{name}: {param.data}")
