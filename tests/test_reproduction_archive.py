import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

from one_d.reproduction_archive import (
    assert_core_audit_non_overlap,
    build_archives,
    collect_archive_entries,
    create_synthetic_archive,
    is_safe_relative_path,
    reserved_doi_metadata,
    sha256_file,
    verify_archive,
    verify_extracted_tree,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_MANIFEST = REPOSITORY_ROOT / "tests/golden/tiny_1d_manifest.json"
EXPECTED_GOLDEN_CONTENT_SHA256 = (
    "91c84e813e5cbfabd0bf0c5be436afc19e64152b7f06c9f1a572a76038108238"
)


def _run_git(repository, *arguments):
    return subprocess.run(
        ["git", *arguments],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _synthetic_repository(tmp_path):
    repository = tmp_path / "repository"
    repository.mkdir()
    _run_git(repository, "init", "-q", "-b", "archive-test")
    _run_git(repository, "config", "user.name", "Archive Test")
    _run_git(repository, "config", "user.email", "archive@example.invalid")

    core = repository / "completed/core"
    audit = repository / "completed/audit"
    core.mkdir(parents=True)
    audit.mkdir(parents=True)
    (core / "tracked.json").write_text('{"value": 1}\n', encoding="utf-8")
    (audit / "report.md").write_text("audit report\n", encoding="utf-8")
    _run_git(repository, "add", "completed/core/tracked.json")
    _run_git(repository, "commit", "-q", "-m", "synthetic source")

    # Deliberately generated/untracked artifacts exercise classification and
    # exclusions without placing production data in the test suite.
    (core / "generated.json").write_text('{"value": 2}\n', encoding="utf-8")
    (core / "__pycache__").mkdir()
    (core / "__pycache__/ignored.pyc").write_bytes(b"cache")
    (audit / "plots_so_far.tar").write_bytes(b"excluded")
    (audit / "machine-path.txt").write_text(
        repository.as_posix() + "/completed/audit/report.md\n", encoding="utf-8"
    )

    commit = _run_git(repository, "rev-parse", "HEAD")
    spec = {
        "schema_version": "1.0.0",
        "validation_profile": "synthetic",
        "archive_date": "20000101",
        "creation_timestamp_utc": "2000-01-01T00:00:00Z",
        "source": {
            "commit": commit,
            "short_commit": commit[:7],
            "branch": "archive-test",
        },
        "runs": {
            "authoritative": "known-run",
            "figures1_3_render": "synthetic-figures1-3",
            "figure4_render": "synthetic-figure4",
            "figure5_render": "synthetic-figure5",
        },
        "scientific_checksums": {},
        "important_files": [],
        "final_figure_checksums": {},
        "global_exclusions": [
            "__pycache__",
            ".pytest_cache",
            ".DS_Store",
            "plots_so_far.tar",
        ],
        "core_groups": [
            {
                "source": "completed/core",
                "destination": "payload/core",
                "role": "authoritative synthetic data",
                "run_id": "{authoritative}",
            }
        ],
        "audit_groups": [
            {
                "source": "completed/audit",
                "destination": "payload/audit",
                "role": "supplemental synthetic audit",
                "run_id": "{authoritative}",
            }
        ],
    }
    return repository, spec


def test_role_classification_non_overlap_and_exclusions(tmp_path):
    repository, spec = _synthetic_repository(tmp_path)
    core = collect_archive_entries(spec, "core", repository_root=repository)
    audit = collect_archive_entries(spec, "audit", repository_root=repository)

    assert_core_audit_non_overlap(core, audit)
    assert {entry.role for entry in core} == {"authoritative synthetic data"}
    assert {entry.authority for entry in core} == {"authoritative"}
    assert {entry.authority for entry in audit} == {"supplemental"}
    status = {Path(entry.source_relative).name: entry.tracked_status for entry in core}
    assert status == {"generated.json": "generated", "tracked.json": "tracked"}
    selected = {entry.source_relative for entry in core + audit}
    assert not any("__pycache__" in path for path in selected)
    assert not any(path.endswith("plots_so_far.tar") for path in selected)
    assert all(is_safe_relative_path(entry.archive_path) for entry in core + audit)
    assert all(not Path(entry.source_relative).is_absolute() for entry in core + audit)


def test_authoritative_run_selection_requires_explicit_permission(tmp_path):
    repository, spec = _synthetic_repository(tmp_path)
    with pytest.raises(ValueError, match="unknown authoritative run ID"):
        collect_archive_entries(
            spec,
            "core",
            repository_root=repository,
            run_overrides={"authoritative": "different-run"},
        )
    entries = collect_archive_entries(
        spec,
        "core",
        repository_root=repository,
        run_overrides={"authoritative": "different-run"},
        allow_unknown_run_ids=True,
    )
    assert {entry.run_id for entry in entries} == {"different-run"}
    with pytest.raises(ValueError, match="unknown archive run key"):
        collect_archive_entries(
            spec,
            "core",
            repository_root=repository,
            run_overrides={"not_a_run": "value"},
            allow_unknown_run_ids=True,
        )


def test_core_audit_overlap_is_rejected(tmp_path):
    repository, spec = _synthetic_repository(tmp_path)
    spec["audit_groups"] = [dict(spec["core_groups"][0])]
    core = collect_archive_entries(spec, "core", repository_root=repository)
    audit = collect_archive_entries(spec, "audit", repository_root=repository)
    with pytest.raises(ValueError, match="overlap"):
        assert_core_audit_non_overlap(core, audit)


def test_synthetic_archive_is_deterministic_and_inventory_is_ordered(tmp_path):
    first = tmp_path / "first.tar.gz"
    second = tmp_path / "second.tar.gz"
    files = {"z/value.bin": b"z", "a/value.bin": b"a"}
    first_report = create_synthetic_archive(first, files)
    second_report = create_synthetic_archive(second, files)
    assert first_report["sha256"] == second_report["sha256"]

    extraction = tmp_path / "verified"
    report = verify_archive(first, extract_to=extraction)
    assert report["safe_extraction"] is True
    inventory = json.loads(
        (extraction / "synthetic_archive/inventory.json").read_text(encoding="utf-8")
    )
    paths = [entry["archive_path"] for entry in inventory["entries"]]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))


def test_checksum_generation_corruption_and_unsafe_path_detection(tmp_path):
    archive = tmp_path / "valid.tar.gz"
    created = create_synthetic_archive(archive, {"payload.bin": b"validated"})
    assert sha256_file(archive) == created["sha256"]

    corrupted = tmp_path / "corrupted.tar.gz"
    data = bytearray(archive.read_bytes())
    data[0] ^= 0xFF
    corrupted.write_bytes(data)
    with pytest.raises(ValueError, match="archive checksum mismatch"):
        verify_archive(corrupted, expected_sha256=created["sha256"])

    unsafe = tmp_path / "unsafe.tar.gz"
    with tarfile.open(unsafe, mode="w:gz") as stream:
        info = tarfile.TarInfo("../escape.txt")
        content = b"escape"
        info.size = len(content)
        stream.addfile(info, io.BytesIO(content))
    with pytest.raises(ValueError, match="unsafe archive member path"):
        verify_archive(unsafe)


def test_missing_and_extra_files_are_detected(tmp_path):
    archive = tmp_path / "valid.tar.gz"
    create_synthetic_archive(archive, {"payload.bin": b"validated"})

    missing_extract = tmp_path / "missing"
    verify_archive(archive, extract_to=missing_extract)
    missing_root = missing_extract / "synthetic_archive"
    (missing_root / "payload.bin").unlink()
    with pytest.raises(ValueError, match="missing files"):
        verify_extracted_tree(missing_root)

    extra_extract = tmp_path / "extra"
    verify_archive(archive, extract_to=extra_extract)
    extra_root = extra_extract / "synthetic_archive"
    (extra_root / "not-in-inventory.txt").write_text("extra\n", encoding="utf-8")
    with pytest.raises(ValueError, match="extra files"):
        verify_extracted_tree(extra_root)


def test_dry_run_writes_nothing_overwrite_refusal_and_portable_text(tmp_path):
    repository, spec = _synthetic_repository(tmp_path)
    output = tmp_path / "archives"
    doi = "10.5281/zenodo.21762243"
    plan = build_archives(
        spec,
        kind="audit",
        repository_root=repository,
        output_directory=output,
        dry_run=True,
        doi=doi,
    )
    assert plan["writes_files"] is False
    assert plan["launches_scientific_execution"] is False
    assert plan["doi"] == {
        "reserved_doi": doi,
        "doi_url": f"https://doi.org/{doi}",
        "doi_status": "reserved_unpublished",
        "repository_record_type": "dataset",
    }
    assert not output.exists()

    built = build_archives(
        spec,
        kind="audit",
        repository_root=repository,
        output_directory=output,
        doi=doi,
    )
    archive = Path(built["archives"][0]["path"])
    assert archive.is_file()
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        build_archives(
            spec,
            kind="audit",
            repository_root=repository,
            output_directory=output,
            doi=doi,
        )

    extraction = tmp_path / "portable"
    verification = verify_archive(archive, extract_to=extraction)
    assert verification["doi"] == plan["doi"]
    archive_root = extraction / "1d_audit_supplement"
    archive_metadata = json.loads(
        (archive_root / "archive_metadata.json").read_text(encoding="utf-8")
    )
    provenance = json.loads(
        (archive_root / "provenance/archive_provenance.json").read_text(
            encoding="utf-8"
        )
    )
    for field, value in plan["doi"].items():
        assert archive_metadata[field] == value
        assert provenance[field] == value
    assert doi in (archive_root / "README.md").read_text(encoding="utf-8")
    assert "depends on the core reproduction archive" in (
        archive_root / "README.md"
    ).read_text(encoding="utf-8")
    assert doi in (archive_root / "CITATION.md").read_text(encoding="utf-8")
    copied_text = (
        extraction
        / "1d_audit_supplement/payload/audit/machine-path.txt"
    ).read_text(encoding="utf-8")
    assert repository.as_posix() not in copied_text
    assert copied_text == "completed/audit/report.md\n"


def test_doi_validation_source_head_resolution_and_clean_build_gate(tmp_path):
    expected = {
        "reserved_doi": "10.5281/zenodo.21762243",
        "doi_url": "https://doi.org/10.5281/zenodo.21762243",
        "doi_status": "reserved_unpublished",
        "repository_record_type": "dataset",
    }
    assert reserved_doi_metadata(expected["reserved_doi"]) == expected
    for invalid in (
        "https://doi.org/10.5281/zenodo.21762243",
        " 10.5281/zenodo.21762243",
        "10.5281/zenodo 21762243",
        "zenodo.21762243",
    ):
        with pytest.raises(ValueError, match="plain ASCII identifier"):
            reserved_doi_metadata(invalid)

    repository, spec = _synthetic_repository(tmp_path)
    commit = _run_git(repository, "rev-parse", "HEAD")
    spec["source"] = {
        "commit": "HEAD",
        "short_commit": "HEAD",
        "branch": "archive-test",
    }
    plan = build_archives(
        spec,
        kind="audit",
        repository_root=repository,
        output_directory=tmp_path / "head-plan",
        dry_run=True,
        doi=expected["reserved_doi"],
    )
    assert plan["source"]["commit"] == commit
    assert commit[:7] in plan["archives"]["audit"]["path"]

    core_build = build_archives(
        spec,
        kind="core",
        repository_root=repository,
        output_directory=tmp_path / "core-doi",
        doi=expected["reserved_doi"],
    )
    core_extract = tmp_path / "core-extract"
    core_verification = verify_archive(
        core_build["archives"][0]["path"], extract_to=core_extract
    )
    assert core_verification["doi"] == expected
    core_root = core_extract / "1d_reproduction"
    assert expected["reserved_doi"] in (core_root / "README.md").read_text(
        encoding="utf-8"
    )
    assert expected["doi_url"] in (core_root / "CITATION.md").read_text(
        encoding="utf-8"
    )

    spec["doi"] = "10.1234/different.record"
    with pytest.raises(ValueError, match="disagrees"):
        build_archives(
            spec,
            kind="audit",
            repository_root=repository,
            output_directory=tmp_path / "conflict",
            dry_run=True,
            doi=expected["reserved_doi"],
        )

    del spec["doi"]
    (repository / "completed/core/tracked.json").write_text(
        '{"value": 3}\n', encoding="utf-8"
    )
    with pytest.raises(ValueError, match="tracked source must be clean"):
        build_archives(
            spec,
            kind="audit",
            repository_root=repository,
            output_directory=tmp_path / "dirty-build",
            doi=expected["reserved_doi"],
        )
    assert not (tmp_path / "dirty-build").exists()


def test_verifier_imports_no_solver_entry_points_and_launches_no_science(tmp_path):
    archive = tmp_path / "valid.tar.gz"
    create_synthetic_archive(archive, {"payload.bin": b"validated"})
    command = (
        "import json,sys; "
        f"sys.path.insert(0, {str(REPOSITORY_ROOT)!r}); "
        "from one_d.reproduction_archive import verify_archive; "
        f"verify_archive({str(archive)!r}); "
        "forbidden={'Nonlinear_Manifold_ROM','one_d.run_fom','one_d.run_rom',"
        "'one_d.operator_inference','one_d.publication_plotting'}; "
        "assert forbidden.isdisjoint(sys.modules); "
        "print(json.dumps({'launches_scientific_execution': False}))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", command],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout)["launches_scientific_execution"] is False


def test_independent_golden_content_checksum_is_unchanged():
    manifest = json.loads(GOLDEN_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["content_checksum"]["sha256"] == EXPECTED_GOLDEN_CONTENT_SHA256
    implementation = (
        REPOSITORY_ROOT / "one_d/reproduction_archive.py"
    ).read_text(encoding="utf-8")
    assert "10.5281/zenodo.21762243" not in implementation
