from __future__ import annotations

import numpy as np
import pytest

from pdac_circuit.generate.evaluate import js_divergence, kmer_spectrum
from pdac_circuit.generate.promoter_gan import samples_to_seqs

def test_kmer_spectrum_normalized():
    spec=kmer_spectrum(["ACGTACGTACGT", "TTTTAAAACCCC"], k=4)
    assert abs(spec.sum() - 1.0) < 1e-9
    assert spec.shape[0] == 256

def test_js_divergence_properties():
    p=np.array([0.5, 0.5])
    q=np.array([0.5, 0.5])
    assert js_divergence(p, q) < 1e-9
    r=np.array([1.0, 0.0])
    assert js_divergence(p, r) > 0.3

def test_samples_to_seqs_roundtrip():
    oh=np.zeros((2, 4, 5), dtype=np.float32)
    for j, b in enumerate([0, 1, 2, 3, 0]):
        oh[0, b, j]=1.0
    for j in range(5):
        oh[1, 3, j]=1.0
    seqs=samples_to_seqs(oh)
    assert seqs == ["ACGTA", "TTTTT"]

@pytest.mark.real_data
@pytest.mark.slow
def test_trained_gan_generates_valid_sequences():

    from pdac_circuit.core.paths import MODELS
    from pdac_circuit.generate.promoter_gan import PromoterGAN

    p=MODELS / "promoter_gan.pt"
    if not p.exists():
        pytest.skip("GAN not trained")
    gan=PromoterGAN.load(p)
    seqs=gan.generate(8, seed=0)
    assert len(seqs) == 8
    for s in seqs:
        assert len(s) == gan.seq_len
        assert set(s) <= set("ACGT")
    assert gan.generate(8, seed=0) == seqs
