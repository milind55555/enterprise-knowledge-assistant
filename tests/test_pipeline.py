from app.ingestion import pipeline as p


def test_clean_text_collapses_whitespace_and_page_footers():
    raw = "Hello   world\r\n\n\n\nPage 3 of 9\nNext"
    assert p.clean_text(raw) == "Hello world\n\nNext"


def test_front_matter_is_parsed_and_removed():
    meta, body = p.parse_front_matter("---\ntitle: T\nallowed_roles: [hr]\n---\nBody")
    assert meta["title"] == "T" and body == "Body"


def test_untagged_documents_fail_closed_to_admin_only():
    doc = p.RawDoc(source="x.md", text="hi", metadata={})
    assert p.tag_document(doc)["allowed_roles"] == ["admin"]


def test_exact_duplicate_removed_and_reported():
    report = p.IngestReport()
    meta = {"title": "t", "department": "d", "allowed_roles": ["all"]}
    a = p.Chunk("1", "Employees get 24 days of leave.", "a.md", **meta)
    b = p.Chunk("2", "Employees get 24 days of leave!", "b.md", **meta)
    assert len(p.deduplicate([a, b], report=report)) == 1
    assert report.exact_duplicates_removed == 1


def test_dedupe_never_hides_content_from_a_wider_audience():
    base = {"title": "t", "department": "d"}
    restricted = p.Chunk("1", "Salary band is 14 to 22 LPA.", "hr.md", allowed_roles=["hr"], **base)
    public = p.Chunk("2", "Salary band is 14 to 22 LPA.", "pub.md", allowed_roles=["all"], **base)
    # public copy must survive even though a restricted copy came first
    assert len(p.deduplicate([restricted, public])) == 2


def test_pipeline_on_sample_docs_removes_the_known_duplicate():
    chunks, report = p.run_pipeline()
    assert report.documents == 9
    assert report.exact_duplicates_removed + report.near_duplicates_removed >= 1
    assert report.chunks_indexed == len(chunks) < report.chunks_created
