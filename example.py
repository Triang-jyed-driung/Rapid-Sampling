"""Run after pip install: python production/example.py."""
import torch
import rapid_sampling as sampler


def main():
    torch.manual_seed(42)
    # For (B,T,V), only logits[:, -1, :] is sampled; scratch storage is (B,V).
    logits = torch.randn(1, 1, 65536, device="cuda", dtype=torch.float32)
    states = sampler.create_rng(seed=123, B=1)
    for _ in range(3):
        token = sampler.sample(
            logits, states, temperature=1., top_k=-1, top_p=.9,
            quantization_bits=32, histogram_bits=[10, 10, 11], block_size=1024,
        )
        print("uint32 sample:", token.item())
    token = sampler.sample(
        logits, states, temperature=1., top_k=-1, top_p=1., quantization_bits=64,
    )
    print("int64 sample:", token.item())
    penalties = torch.zeros((1, 65536), device="cuda")
    token = sampler.sample_with_penalties(
        logits, penalties, states, presence_penalty=.75, repetition_penalty=.25,
        penalty_decay=.5, temperature=1., top_k=50, top_p=.9,
    )
    print("sample with updated penalties:", token.item())


if __name__ == "__main__":
    main()
