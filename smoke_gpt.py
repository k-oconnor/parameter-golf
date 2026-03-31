#!/usr/bin/env python3
"""Quick smoke tests: RBL package + tiny GPT (flash / RBL) forward and backward. No dataset required.

For real FineWeb val tokens + SentencePiece, run: python smoke_train_fineweb.py
"""

from __future__ import annotations

import sys


def main() -> int:
    import torch

    import train_gpt

    # Config parsing (CPU)
    assert train_gpt.RBL_OPS_HEAD_TYPES == ("always", "eventually", "until", "next")
    cfg4 = train_gpt.parse_rbl_head_config(4, "always,eventually,until,next")
    assert cfg4 == list(train_gpt.RBL_OPS_HEAD_TYPES)
    try:
        train_gpt.parse_rbl_head_config(4, "")
    except ValueError:
        pass
    else:
        print("FAIL: expected ValueError for empty RBL_HEAD_CONFIG in parse_rbl_head_config")
        return 1
    try:
        train_gpt.parse_rbl_head_config(4, "always")
    except ValueError:
        pass
    else:
        print("FAIL: expected ValueError for short RBL_HEAD_CONFIG")
        return 1
    print("ok parse_rbl_head_config")

    if not torch.cuda.is_available():
        print("skip_cuda: no CUDA — CPU checks only")
        return 0

    device = torch.device("cuda:0")
    torch.manual_seed(0)
    seq = 64
    bsz = 2
    vocab = 1024
    x = torch.randint(0, vocab, (bsz, seq), device=device)
    y = torch.randint(0, vocab, (bsz, seq), device=device)

    def run_flash() -> None:
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
            train_seq_len=seq,
            rbl_head_config=None,
        ).to(device=device, dtype=torch.bfloat16)
        m.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = m(x, y)
        loss.backward()
        print(f"ok flash forward+backward loss={loss.item():.4f}")

    def run_rbl() -> None:
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
            train_seq_len=seq,
            rbl_head_config=heads,
        ).to(device=device, dtype=torch.bfloat16)
        m.train()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            loss = m(x, y)
        loss.backward()
        print(f"ok rbl forward+backward loss={loss.item():.4f}")

    run_flash()
    run_rbl()
    return 0


if __name__ == "__main__":
    sys.exit(main())
