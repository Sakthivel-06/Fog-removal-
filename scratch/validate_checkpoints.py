import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import torch
from novel_method.learned_residual.tiny_unet import TinyUNet

def validate_checkpoint(ckpt_path="checkpoints/delta_t_tiny_unet_best.pt"):
    print(f"=== Phase 5: Checkpoint Validation ({ckpt_path}) ===")
    assert os.path.exists(ckpt_path), f"Checkpoint {ckpt_path} not found!"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    
    model = TinyUNet().to("cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    # 1. Parameter count
    p_count = model.count_parameters()
    print(f"1. Exact Parameter Count: {p_count:,} (Expected: 13,921)")
    assert p_count == 13921, f"Parameter count mismatch! {p_count}"
    
    # 2. Forward pass check
    x = torch.randn(2, 8, 135, 240)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
        
    print(f"2. Output Shape: {tuple(out1.shape)}")
    print(f"3. Delta-t Range: [{out1.min().item():.4f}, {out1.max().item():.4f}]")
    print(f"4. NaN/Inf Check: NaN={torch.isnan(out1).any().item()}, Inf={torch.isinf(out1).any().item()}")
    print(f"5. Deterministic Check (diff between identical runs): {torch.max(torch.abs(out1 - out2)).item():.8f}")
    
    assert not torch.isnan(out1).any(), "NaN detected!"
    assert out1.min().item() >= -0.3001 and out1.max().item() <= 0.3001, f"Out of bounds! [{out1.min().item()}, {out1.max().item()}]"
    assert torch.max(torch.abs(out1 - out2)).item() == 0.0, "Non-deterministic output!"
    print("=== Phase 5 Checkpoint Validation PASSED ===")

if __name__ == '__main__':
    validate_checkpoint("checkpoints/delta_t_tiny_unet_best.pt")
    validate_checkpoint("checkpoints/delta_t_tiny_unet_no_temp.pt")
