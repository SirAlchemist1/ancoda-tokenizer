"""Downloader rules that do not touch the Hub."""

from pathlib import Path

from ancoda_tok.download import (
    MANIFEST_KEYS,
    aya_kept,
    budget_decision,
    code_license_kept,
    devanagari_letter_share,
    iter_plain_documents,
    make_doc_id,
    room_ok,
    script_decision,
    split_plain_documents,
    utf8_len,
    write_table,
)


def test_budget_stops_before_the_next_document() -> None:
    assert budget_decision(8, 4, 10) == "stop"
    assert budget_decision(6, 4, 10) == "take"
    assert budget_decision(0, 15, 10) == "take"
    assert budget_decision(0, 4, None) == "take"


def test_doc_id_is_stable() -> None:
    assert make_doc_id("fw2_npi", "data/npi_Deva/train/a.parquet", 3) == (
        "fw2_npi\x1fdata/npi_Deva/train/a.parquet\x1f3"
    )


def test_room_uses_the_50gb_floor() -> None:
    assert room_ok(100, 40, floor=50)
    assert not room_ok(100, 60, floor=50)


def test_plain_documents_follow_blank_lines() -> None:
    docs, rule = split_plain_documents(["a", "", "b"])
    assert rule == "blank-lines"
    assert docs == ["a", "b"]
    docs, rule = split_plain_documents(["a", "b"])
    assert rule == "lines"
    assert docs == ["a", "b"]
    docs, rule = split_plain_documents(["a", "  ", "b"])
    assert rule == "blank-lines"
    assert docs == ["a", "b"]


def test_plain_reader_matches_and_resumes(tmp_path: Path) -> None:
    path = tmp_path / "ne.txt"
    path.write_bytes(b"a\n\nb\n")
    first = list(iter_plain_documents(path, 0, 0, True))
    assert [item.payload for item in first] == ["a", "b"]
    rest = list(iter_plain_documents(path, first[0].after, 1, True))
    assert [item.payload for item in rest] == ["b"]


def test_devanagari_decision() -> None:
    assert devanagari_letter_share("नमस्ते") == 1
    assert devanagari_letter_share("hello") == 0
    assert script_decision(0.8) == "keep"
    assert script_decision(0.1) == "skip"
    assert script_decision(0.3) == "stop"
    assert script_decision(None) == "stop"


def test_code_and_aya_filters() -> None:
    assert code_license_kept(" MIT ") == "mit"
    assert code_license_kept("gpl-3.0") is None
    assert aya_kept({"language_code": "nep", "language": "Hindi"})
    assert aya_kept({"language_code": "npi", "language": "Nepali"})
    assert not aya_kept({"language_code": "hin", "language": "Hindi"})


def test_shard_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "shard-00000.parquet"
    rows = [
        {"doc_id": "s\x1fp\x1f0", "text": "क", "source": "s", "license": "odc-by-1.0"},
        {"doc_id": "s\x1fp\x1f1", "text": "kh", "source": "s", "license": "odc-by-1.0"},
    ]
    write_table(path, rows)
    import pyarrow.parquet as pq

    table = pq.read_table(path)
    assert table.column_names == ["doc_id", "text", "source", "license"]
    assert table.column("text").to_pylist() == ["क", "kh"]
    assert utf8_len("क") == 3


def test_release_deletes_an_unlinked_hub_blob(tmp_path: Path, monkeypatch) -> None:
    from ancoda_tok.download import release_download

    monkeypatch.setenv("HF_HOME", str(tmp_path))
    blob = tmp_path / "hub" / "blobs" / "0d" / "0ddde87b"
    blob.parent.mkdir(parents=True)
    blob.write_bytes(b"parquet")
    repo = tmp_path / "hub" / "datasets--org--name"
    pointer = repo / "blobs" / "etag"
    pointer.parent.mkdir(parents=True)
    pointer.symlink_to(blob)
    link = repo / "snapshots" / "rev" / "file.parquet"
    link.parent.mkdir(parents=True)
    link.symlink_to(blob)
    other = repo / "snapshots" / "rev" / "keep.parquet"
    other.symlink_to(blob)
    release_download(link, repo)
    assert not link.exists()
    assert blob.exists()
    assert pointer.exists()
    release_download(other, repo)
    assert not blob.exists()
    assert not pointer.exists()


def test_manifest_keys_cover_the_required_record() -> None:
    required = {
        "input_sha256",
        "config_sha256",
        "git_commit",
        "library_versions",
        "unicode_segmentation",
        "unicode",
        "seed",
        "started_at",
        "finished_at",
        "hf_revision",
    }
    assert required <= set(MANIFEST_KEYS)
