"""Byte BPE specials, ARC color tokens, freeze artifact (spec 3.1, 7)."""

from __future__ import annotations

from tokenizer import ARC_ROW_ID, COLOR_BASE, SPECIALS, Tokenizer, train


def test_bpe_roundtrip_and_grid_colors(tmp_path):
    tok = train(["hello hello world", "hello"], n_merges=8)
    ids = tok.encode("hello")
    assert tok.decode(ids) == "hello"
    assert tok.vocab_size >= 256
    grid = [[1, 2, 3], [4, 5, 0]]
    gids = tok.encode_grid(grid)
    assert ARC_ROW_ID in gids
    assert all((COLOR_BASE <= i < COLOR_BASE + 10) or i == ARC_ROW_ID for i in gids)
    assert tok.decode_grid(gids) == grid
    assert tok.color_ids[9] == COLOR_BASE + 9
    assert "<bos>" in tok.specials
    assert tok.specials["<pad>"] == SPECIALS["<pad>"]
    path = tmp_path / "vocab.json"
    tok.freeze(path)
    assert path.is_file()
    loaded = Tokenizer.load(path)
    assert loaded.encode("hello") == ids
    assert loaded.decode(ids) == "hello"
    assert loaded.decode_grid(loaded.encode_grid(grid)) == grid
