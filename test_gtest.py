"""Likelihood-ratio goodness-of-fit tests against an independent CPU reference.

Install scipy, then run: python -m unittest discover -s production -p test_gtest.py
CPU reference/statistical arithmetic uses float64; device sampling uses no FP64.
"""
import math
import unittest

import torch
from scipy.stats import chi2
import rapid_sampling as sampler


def reference(logits, top_k=-1, top_p=1., temperature=1.):
    """Fractional top-k/top-p cutoff, shared equally by all boundary ties."""
    probabilities = torch.softmax(logits.double() / temperature, dim=0)
    k = len(logits) if top_k <= 0 else min(top_k, len(logits))
    if top_p == 0.:
        k, top_p = 1, 1.
    result = torch.zeros_like(probabilities)
    count, mass = 0, 0.
    for value in sorted(set(probabilities.tolist()), reverse=True):
        if value == 0.:
            continue
        mask = probabilities == value
        tied = int(mask.sum())
        fraction = max(0., min(1., (k-count)/tied,
                              (top_p-mass)/(value*tied) if top_p < 1. else 1.))
        result[mask] = value*fraction
        count += tied
        mass += value*tied
    return result / result.sum()


def g_test(counts, probabilities):
    """Pool sparse bins to expected count >=10 before applying chi-square asymptotics."""
    counts = counts.double().cpu()
    probabilities = probabilities.double().cpu()
    if counts[probabilities == 0].sum():
        raise AssertionError("sampled a category with zero reference probability")
    expected = probabilities * counts.sum()
    keep = expected >= 10.
    observed = counts[keep].tolist()
    expected_bins = expected[keep].tolist()
    tail_expected = expected[~keep].sum().item()
    tail_observed = counts[~keep].sum().item()
    if tail_expected:
        if tail_expected >= 10. or not observed:
            observed.append(tail_observed)
            expected_bins.append(tail_expected)
        else:
            index = min(range(len(expected_bins)), key=expected_bins.__getitem__)
            observed[index] += tail_observed
            expected_bins[index] += tail_expected
    if len(observed) < 2:
        raise AssertionError("G-test needs at least two populated categories")
    statistic = 2*sum(o*math.log(o/e) for o, e in zip(observed, expected_bins) if o)
    degrees = len(observed)-1  # No parameters are fitted to the observations.
    return statistic, degrees, float(chi2.sf(statistic, degrees))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU unavailable")
class SamplingGTests(unittest.TestCase):
    # Bonferroni: fewer than 64 comparisons, family false-rejection probability <=1e-5.
    alpha = 1e-5/64

    def check_fit(self, label, counts, expected):
        statistic, degrees, p = g_test(counts, expected)
        print(f"{label}: G={statistic:.5f}, df={degrees}, p={p:.6g}", flush=True)
        self.assertGreater(p, self.alpha, label)

    def test_batched_uint32_and_int64(self):
        row = torch.tensor([2., 2., 1., .5, 0., -1., -2., -20., -80.])
        n = 16384
        logits = row.cuda().expand(n, -1).contiguous()
        cases = [(-1, 1., 1.), (3, .8, .7), (1, 1., 1.), (-1, .4, 1.),
                 (2, .1, 1.), (-1, 0., 1.)]
        for bits in (32, 64):
            for block in (256, 512, 1024):
                for k, p, temperature in cases:
                    with self.subTest(bits=bits, block=block, k=k, p=p, temperature=temperature):
                        samples = sampler.sample(
                            logits, sampler.create_rng(7241, n), temperature, k, p,
                            quantization_bits=bits, block_size=block,
                        )
                        counts = torch.bincount(samples.long(), minlength=len(row))
                        self.check_fit(f"int{bits} block={block} k={k} p={p}", counts,
                                       reference(row, k, p, temperature))

    def test_cooperative_cross_tile_ties(self):
        indices = torch.tensor([0, 2048, 1, 2049, 4096], device="cuda")
        support = torch.tensor([4., 3.1, 2., 2., 2.])
        logits = torch.full((1, 65536), -1000., device="cuda")
        logits[0, indices] = support.cuda()
        n = 4096
        for block in (512, 1024):
            for k, p in ((-1, 1.), (3, 1.), (-1, .9)):
                with self.subTest(block=block, k=k, p=p):
                    states = sampler.create_rng(92817, 1)
                    samples = torch.cat([
                        sampler.sample(
                            logits, states, 1., k, p, block_size=block,
                        ) for _ in range(n)
                    ])
                    counts = torch.bincount(samples.long(), minlength=65536)
                    self.assertEqual(int(counts.sum()-counts[indices].sum()), 0)
                    self.check_fit(f"cooperative block={block} k={k} p={p}",
                                   counts[indices], reference(support, k, p))

    def test_cooperative_candidate_overflow(self):
        logits = torch.zeros((1, 65536), device="cuda")
        logits[0, 0] = math.log(65536)
        n = 4096
        # A head plus 16 spatial tail groups detects biased tile selection.
        expected = torch.tensor([65536., *([4096.]*16)], dtype=torch.float64)
        expected[1] -= 1
        expected /= 65536+32767
        expected[1:] *= 32767/65535
        expected /= expected.sum()
        for block in (512, 1024):
            states = sampler.create_rng(81272, 1)
            samples = torch.cat([
                sampler.sample(
                    logits, states, 1., 32768, 1., block_size=block,
                ) for _ in range(n)
            ]).long()
            categories = torch.where(samples == 0, 0, 1+samples//4096)
            self.check_fit(f"candidate overflow block={block}",
                           torch.bincount(categories, minlength=17), expected)


if __name__ == "__main__":
    unittest.main()
