import torch

def load_model(path, model) -> torch.nn.Module:
    model.load_state_dict(torch.load(path))
    model.eval()
    return model

def inference(data, model):
    pred = model(data)
    for p, y in zip(pred, data.y):
        print(p, y)
    return pred
