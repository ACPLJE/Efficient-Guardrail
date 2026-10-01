"""purge_sparse_incomplete: full-size (sparse, xet-written) partial blobs go, short partials stay."""
from pathlib import Path

from lgs1.download import purge_sparse_incomplete


def test_purge_only_full_size_partials(tmp_path: Path):
    blobs = tmp_path / "blobs"
    blobs.mkdir()
    full = blobs / ("a" * 64 + ".incomplete")          # xet: sparse file already at target size
    short = blobs / ("b" * 64 + ".1f2e3d4c.incomplete")  # classic: genuinely partial -> resumable
    unknown = blobs / ("c" * 64 + ".incomplete")       # not an LFS file we know -> untouched
    done = blobs / ("d" * 64)                           # finished blob -> untouched
    with full.open("wb") as f:
        f.truncate(1000)
    short.write_bytes(b"x" * 10)
    unknown.write_bytes(b"y" * 5)
    done.write_bytes(b"z" * 20)
    removed = purge_sparse_incomplete(blobs, {"a" * 64: 1000, "b" * 64: 1000, "d" * 64: 20})
    assert removed == [full]
    assert not full.exists() and short.exists() and unknown.exists() and done.exists()


def test_purge_missing_dir_is_noop(tmp_path: Path):
    assert purge_sparse_incomplete(tmp_path / "nope", {"a" * 64: 1}) == []
