# Tests for model correctness.
import torch
from rashomon_emotion.model import EEGGNN

def test_model_forward():
    model = EEGGNN(input_dim=10, hidden_dim=5, output_dim=2)
    x = torch.randn(4, 10)  # batch of 4 samples
    out = model(x)
    assert out.shape == (4, 2)
    assert torch.is_tensor(out)
