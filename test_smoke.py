"""Run after installation: python -m unittest discover -s production -p test_smoke.py."""
import math
import unittest

import torch
import rapid_sampling

LAYOUTS = [
    (11, 10, 10), (10, 11, 10), (10, 10, 11), (9, 11, 11), (11, 9, 11),
    (10, 8, 9, 4), (8, 8, 8, 7), (9, 8, 7, 7), (7, 8, 8, 8),
    (11, 8, 8, 4), (10, 7, 7, 7), (8, 9, 9, 5),
    (7, 6, 6, 6, 6), (8, 6, 6, 6, 5), (9, 6, 6, 6, 4),
    (6, 6, 6, 6, 7), (10, 6, 5, 5, 5),
]


@unittest.skipUnless(torch.cuda.is_available(), "CUDA GPU unavailable")
class CudaSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampler = rapid_sampling

    def draw(self, logits, bits, top_k=-1, top_p=1.0, temperature=1.0, seed=123, block_size=1024):
        batch = logits.shape[0] if logits.ndim > 1 else 1
        return self.sampler.sample(
            logits, self.sampler.create_rng(seed, batch), temperature,
            top_k, top_p, quantization_bits=bits, block_size=block_size,
        )

    def test_last_timestep_and_vocabulary_sizes(self):
        for bits in (32, 64):
            for vocab in (1, 3, 1024, 1048576):
                logits = torch.full((2, 3, vocab), -1000.0, device="cuda")
                logits[:, 0, 0] = 1000
                logits[:, -1, -1] = 1000
                out = self.draw(logits, bits)
                self.assertEqual(out.tolist(), [vocab - 1] * 2)

    def test_seed_reproducibility_and_state_advance(self):
        logits = torch.zeros((512, 37), device="cuda")
        for bits in (32, 64):
            states = self.sampler.create_rng(19, 512)
            first = self.sampler.sample(logits, states, 1., -1, 1., bits)
            second = self.sampler.sample(logits, states, 1., -1, 1., bits)
            self.assertTrue(torch.equal(first, self.draw(logits, bits, seed=19)))
            self.assertFalse(torch.equal(first, second))

    def test_filtered_distribution(self):
        # Reference implements fractional boundary compensation, including ties,
        # rather than the usual convention of retaining the entire last token.
        row = torch.tensor([2., 2., 1., .5, 0., -1., -2., -20., -80.])
        for bits in (32, 64):
            for top_k, top_p, temp in ((-1, 1., 1.), (3, .8, .7), (1, 1., 1.), (-1, .4, 1.)):
                probs = torch.softmax(row / temp, 0).double()
                filtered = torch.zeros_like(probs)
                mass, count = 0., 0
                k = len(row) if top_k < 0 else top_k
                for p in sorted(set(probs.tolist()), reverse=True):
                    if p == 0.:
                        continue
                    mask = probs == p
                    n = int(mask.sum())
                    comp = max(0., min(1., (k - count) / n, (top_p - mass) / (p * n)))
                    filtered[mask] = p * comp
                    mass += p * n
                    count += n
                target = 4.26e9 if bits == 32 else 2**62
                quantized = torch.floor(filtered * (target / filtered.sum()))
                expected = quantized / quantized.sum()
                batch = 8192
                out = self.draw(row.cuda().expand(batch, -1).contiguous(), bits, top_k, top_p, temp)
                observed = torch.bincount(out.long(), minlength=len(row)).cpu().double() / batch
                self.assertEqual(observed[expected == 0].sum().item(), 0.)
                sigma = torch.sqrt(expected * (1 - expected) / batch)
                self.assertTrue(torch.all((observed - expected).abs() < 6 * sigma + .002))

    def test_penalties(self):
        for bits in (32, 64):
            logits = torch.tensor([[100., 0., -100.]], device="cuda")
            penalties = torch.tensor([[0., 2., 3.]], device="cuda")
            out = self.sampler.sample_with_penalties(
                logits, penalties, self.sampler.create_rng(1, 1),
                .75, .25, .5, 1., 1, 1., bits,
            )
            self.assertEqual(out.item(), 0)
            torch.testing.assert_close(penalties, torch.tensor([[.75, 1., 1.5]], device="cuda"))
            self.sampler.sample_with_penalties(
                logits, penalties, self.sampler.create_rng(1, 1),
                .75, .25, .5, 1., 1, 1., bits,
            )
            self.assertAlmostEqual(penalties[0, 0].item(), .625)

    def test_uniform_ties_and_zero_top_p(self):
        for bits in (32, 64):
            logits = torch.zeros((4096, 7), device="cuda")
            for k, p in ((2, .1), (-1, 0.)):
                out = self.draw(logits, bits, k, p)
                counts = torch.bincount(out.long(), minlength=7)
                self.assertTrue(torch.all((counts - 4096 / 7).abs() < 100))

    def test_histogram_layouts(self):
        torch.manual_seed(42)
        for bits in (32, 64):
            for uniform in (False, True):
                logits = torch.randn((64, 513), device="cuda") * 3
                if uniform:
                    logits.zero_()
                for k, p in ((-1, .9), (17, .3)):
                    expected = self.draw(logits, bits, k, p)
                    for layout in LAYOUTS:
                        actual = self.sampler.sample(
                            logits, self.sampler.create_rng(123, 64), 1., k, p,
                            bits, histogram_bits=layout,
                        )
                        self.assertTrue(torch.equal(expected, actual), str(layout))
            logits = torch.randn((4, 513), device="cuda")
            for layout in LAYOUTS:
                penalties = torch.zeros_like(logits)
                expected = self.draw(logits, bits, 17, .9)
                actual = self.sampler.sample_with_penalties(
                    logits, penalties, self.sampler.create_rng(123, 4),
                    .75, .25, .5, 1., 17, .9, bits, layout,
                )
                self.assertTrue(torch.equal(expected, actual), str(layout))
                self.assertEqual((penalties != 0).sum().item(), 4)

    def test_invalid_histogram_layouts(self):
        logits = torch.zeros(4, device="cuda")
        states = self.sampler.create_rng(123, 1)
        for layout in ([10, 10], [12, 10, 9], [8, 8, 8], [0, 10, 10, 11]):
            with self.assertRaisesRegex(RuntimeError, "histogram"):
                self.sampler.sample(
                    logits, states, 1., -1, .9, 32, layout,
                )

    def test_block_sizes(self):
        for size in (256, 512, 1024):
            for bits in (32, 64):
                logits = torch.full((2, 3, 65536), -1000., device="cuda")
                logits[:, -1, -1] = 1000.
                self.assertEqual(self.draw(logits, bits, block_size=size).tolist(), [65535, 65535])
                logits = torch.zeros((8192, 7), device="cuda")
                out = self.draw(logits, bits, 2, .1, block_size=size)
                counts = torch.bincount(out.long(), minlength=7)
                self.assertTrue(torch.all((counts - 8192 / 7).abs() < 150))

    def test_invalid_block_size(self):
        with self.assertRaisesRegex(RuntimeError, "block_size"):
            self.draw(torch.zeros(4, device="cuda"), 32, block_size=128)

    def test_invalid_quantization(self):
        with self.assertRaisesRegex(RuntimeError, "quantization_bits"):
            self.draw(torch.zeros(4, device="cuda"), 16)



@unittest.skipUnless(torch.cuda.is_available(), 'CUDA GPU unavailable')
class CooperativeSamplingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sampler = rapid_sampling

    def sample_many(self, logits, k, p, n=2048):
        states = self.sampler.create_rng(123, 1)
        return torch.cat([self.sampler.sample(logits, states, 1., k, p) for _ in range(n)])

    def test_last_timestep_rng_and_graph(self):
        logits = torch.full((1, 3, 65536), -1000., device='cuda')
        logits[0, 0, 0] = 1000.
        logits[0, -1, -1] = 1000.
        states = self.sampler.create_rng(123, 1)
        out = self.sampler.sample(logits, states, 1., -1, .9)
        self.assertEqual(out.item(), 65535)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = self.sampler.sample(logits, states, 1., -1, .9)
        graph.replay()
        self.assertEqual(out.item(), 65535)
        logits.zero_()
        first = self.sampler.sample(logits, states, 1., -1, 1.)
        second = self.sampler.sample(logits, states, 1., -1, 1.)
        self.assertNotEqual(first.item(), second.item())

    def test_cross_tile_ties(self):
        logits = torch.full((1, 65536), -1000., device='cuda')
        logits[0, 0] = 4.; logits[0, 2048] = 3.1
        tied = [1, 2049, 4096]
        logits[0, tied] = 2.
        samples = self.sample_many(logits, 3, 1.)
        indices = torch.tensor([0, 2048, *tied], device='cuda')
        counts = torch.bincount(samples.long(), minlength=65536)[indices].float() / len(samples)
        expected = torch.tensor([math.exp(4), math.exp(3.1), *([math.exp(2)/3]*3)], device='cuda')
        expected /= expected.sum()
        sigma = torch.sqrt(expected*(1-expected)/len(samples))
        self.assertTrue(torch.all((counts-expected).abs() < 6*sigma+.002))
        self.assertTrue(torch.all(counts[2:] > 0))

    def test_candidate_cache_overflow(self):
        logits = torch.zeros((1, 65536), device='cuda')
        logits[0, 0] = math.log(65536)
        samples = self.sample_many(logits, 32768, 1.)
        actual = (samples == 0).float().mean().item()
        expected = 65536 / (65536 + 32767)
        self.assertLess(abs(actual-expected), 6*math.sqrt(expected*(1-expected)/len(samples))+.002)

    def test_all_layouts_last_timestep(self):
        generator = torch.Generator().manual_seed(42)
        logits = (torch.randn((1, 2, 65536), generator=generator) * 3).cuda()
        for k,p in [(-1,.9), (17,.3), (-1,1.)]:
            reference = None
            for layout in LAYOUTS:
                states = self.sampler.create_rng(123, 1)
                out = self.sampler.sample(logits, states, 1., k, p, histogram_bits=layout)
                if reference is None: reference = out
                self.assertTrue(torch.equal(out, reference), str(layout))


if __name__ == "__main__":
    unittest.main()
