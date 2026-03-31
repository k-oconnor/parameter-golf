#!/usr/bin/env python3
"""Load real FineWeb val tokens + tokenizer; one forward/backward (flash + RBL). Fast; no full training loop."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    import torch
    import sentencepiece as spm

    import train_gpt

    root = Path(__file__).resolve().parent
    val_pattern = str(root / "data" / "datasets" / "fineweb10B_sp1024" / "fineweb_val_*.bin")
    tok_path = root / "data" / "tokenizers" / "fineweb_1024_bpe.model"
    if not tok_path.is_file():
        print(f"skip: missing tokenizer {tok_path}")
        return 0
    from glob import glob as glob_fn

    if not glob_fn(val_pattern):
        print("skip: no fineweb_val_*.bin under data/datasets/fineweb10B_sp1024")
        return 0

    # Windows PyTorch often has no Flash SDPA; allow math fallback like train_gpt + SDP_ALLOW_MATH=1.
    torch.backends.cuda.matmul.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(True)

    if not torch.cuda.is_available():
        print("skip_cuda: no GPU")
        return 0
    device = torch.device("cuda:0")

    sp = spm.SentencePieceProcessor(model_file=str(tok_path))
    vocab = int(sp.vocab_size())
    seq_len = 1024
    tokens = train_gpt.load_validation_tokens(val_pattern, seq_len)
    local = tokens[: seq_len + 1].to(device=device, dtype=torch.int64)
    x = local[:-1].unsqueeze(0)
    y = local[1:].unsqueeze(0)

    def run_one(attn: str) -> None:
        if attn == "flash":
            m = train_gpt.GPT(
                vocab_size=vocab,
                num_layers=2,
                model_dim=256,
                num_heads=8,
                num_kv_heads=4,
                mlp_mult=2,
                tie_embeddings=True,
                tied_embed_init_std=0.02,
                logit_softcap=30.0,
                rope_base=10000.0,
                qk_gain_init=1.5,
                attention_backend="flash",
                train_seq_len=seq_len,
                rbl_head_config=None,
            )
        else:
            heads = list(train_gpt.RBL_OPS_HEAD_TYPES)
            m = train_gpt.GPT(
                vocab_size=vocab,
                num_layers=2,
                model_dim=256,
                num_heads=4,
                num_kv_heads=4,
                mlp_mult=2,
                tie_embeddings=True,
                tied_embed_init_std=0.02,
                logit_softcap=30.0,
                rope_base=10000.0,
                qk_gain_init=1.5,
                attention_backend="rbl",
                train_seq_len=seq_len,
                rbl_head_config=heads,
            )
        m = m.to(device=device, dtype=torch.bfloat16)
        m.train()
        for mod in m.modules():
            if isinstance(mod, train_gpt.CastedLinear):
                mod.float()
        train_gpt.restore_low_dim_params_to_fp32(m)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = m(x, y)
        loss.backward()
        print(f"ok {attn} loss={float(loss):.4f} (real val slice, vocab={vocab})")

    run_one("flash")
    m = None
    torch.cuda.empty_cache()
    run_one("rbl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
