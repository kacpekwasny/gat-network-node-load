import torch

def load_model(path, model) -> torch.nn.Module:
    model.load_state_dict(torch.load(path))
    model.eval()
    return model