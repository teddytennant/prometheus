"""BPE roundtrip (B4)."""

from tokenizer import train


def test_roundtrip():
    tok = train(["hello hello world", "hello"], n_merges=8)
    ids = tok.encode("hello")
    assert tok.decode(ids) == "hello"
    assert tok.vocab_size >= 256
