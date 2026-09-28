import numpy as np
import pytest

from wavecaster.fec import LDPC, RSCode, ReedSolomonError, get_codec
from wavecaster.fec.ldpc import gf2_rref, parse_alist, peg_matrix, to_alist

RNG = np.random.default_rng(1234)


def awgn_llr(cw, ebn0_db, rate):
    sigma = np.sqrt(1 / (2 * rate * 10 ** (ebn0_db / 10)))
    y = (1 - 2.0 * cw) + sigma * RNG.standard_normal(len(cw))
    return 2 * y / sigma ** 2


@pytest.mark.parametrize("spec", ["none", "hamming74", "rs:255,223", "rs:40,30", "ldpc:256,0.5",
                                  "ldpc:240,0.75,3,7", "turbo:128,6", "turbo:128,6,p", "conv:64"])
def test_noiseless_roundtrip_stream(spec):
    c = get_codec(spec)
    bits = RNG.integers(0, 2, 3 * c.k + 5).astype(np.uint8)  # non-multiple of k
    coded = c.encode(bits)
    assert len(coded) == c.encoded_length(len(bits))
    res = c.decode(8.0 * (1 - 2.0 * coded), len(bits))
    assert res.ok and np.array_equal(res.bits, bits)


@pytest.mark.parametrize("n,k", [(255, 223), (255, 239), (60, 40)])
def test_rs_errors_and_erasures_at_bound(n, k):
    code = RSCode(n, k)
    for _ in range(20):
        msg = RNG.integers(0, 256, k, dtype=np.uint8)
        cw = code.encode(msg)
        ne = int(RNG.integers(0, (n - k) // 2 + 1))
        nf = (n - k) - 2 * ne
        pos = RNG.choice(n, ne + nf, replace=False)
        r = cw.copy()
        r[pos] ^= RNG.integers(1, 256, ne + nf, dtype=np.uint8)
        dec, nc = code.decode(r, pos[ne:])
        assert np.array_equal(dec, msg) and nc == ne + nf


def test_rs_detects_uncorrectable():
    code = RSCode(255, 223)
    cw = code.encode(RNG.integers(0, 256, 223, dtype=np.uint8))
    cw[:20] ^= 0x5A
    with pytest.raises(ReedSolomonError):
        code.decode(cw)


def test_rs_gmd_uses_soft_reliability():
    c = get_codec("rs:255,239")  # t = 8 symbol errors
    bits = RNG.integers(0, 2, c.k).astype(np.uint8)
    llr = 6.0 * (1 - 2.0 * c.encode_block(bits))
    # corrupt 12 bytes (> t) but mark them unreliable: erasure decoding recovers
    for b in RNG.choice(255, 12, replace=False):
        llr[b * 8] = -0.1 * np.sign(llr[b * 8])
    res = c.decode_block(llr)
    assert res.ok and np.array_equal(res.bits, bits)


def test_ldpc_structure():
    H = peg_matrix(200, 100, 3, seed=3)
    assert (H.sum(axis=0) == 3).all()
    O = H.astype(int).T @ H.astype(int)
    np.fill_diagonal(O, 0)
    assert (O <= 1).all(), "PEG code should be free of 4-cycles"
    assert np.array_equal(parse_alist(to_alist(H)), H)
    code = LDPC(H)
    cw = code.encode_block(RNG.integers(0, 2, code.k).astype(np.uint8))
    assert not code.syndrome(cw).any()


def test_ldpc_rank_deficient_matrix():
    H = peg_matrix(64, 32, 3, seed=0)
    H = np.vstack([H, H[0] ^ H[1]])  # dependent row
    R, piv = gf2_rref(H)
    code = LDPC(H)
    assert code.k == 64 - len(piv)
    cw = code.encode_block(RNG.integers(0, 2, code.k).astype(np.uint8))
    assert not code.syndrome(cw).any()


@pytest.mark.parametrize("spec,ebn0,max_fer", [
    ("ldpc:512,0.5", 3.0, 0.1),
    ("turbo:256,8", 2.0, 0.1),
    ("conv:128", 4.5, 0.1),
])
def test_coding_gain_over_awgn(spec, ebn0, max_fer):
    c = get_codec(spec)
    fails = 0
    for _ in range(20):
        m = RNG.integers(0, 2, c.k).astype(np.uint8)
        fails += not np.array_equal(c.decode_block(awgn_llr(c.encode_block(m), ebn0, c.rate)).bits, m)
    assert fails / 20 <= max_fer
