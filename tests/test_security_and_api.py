import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
EMP = {"X-API-Key": "demo-employee-key"}
HR = {"X-API-Key": "demo-hr-key"}
ADMIN = {"X-API-Key": "demo-admin-key"}


@pytest.fixture(scope="module", autouse=True)
def _ingest():
    assert client.post("/ingest", headers=ADMIN).status_code == 200


def ask(headers, q, **kw):
    return client.post("/query", json={"question": q, **kw}, headers=headers)


def test_health_needs_no_key():
    assert client.get("/health").json()["status"] == "ok"


def test_missing_and_invalid_keys_are_rejected():
    assert ask({}, "How many days of annual leave?").status_code == 401
    assert ask({"X-API-Key": "nope"}, "How many days of annual leave?").status_code == 401


def test_answer_has_citation_and_sources():
    r = ask(EMP, "How many days of paid annual leave do I get each year?").json()
    assert "24" in r["answer"] and not r["abstained"]
    assert any(s["source"] == "leave_policy.md" for s in r["sources"])


def test_employee_cannot_read_hr_documents():
    r = ask(EMP, "What is the salary band for level L3?").json()
    assert r["abstained"]
    assert all(s["source"] != "hr_compensation_confidential.md" for s in r["sources"])
    assert "14 to 22" not in r["answer"]


def test_hr_can_read_hr_but_not_finance_documents():
    assert "14 to 22" in ask(HR, "What is the salary band for level L3?").json()["answer"]
    r = ask(HR, "What is the FY2026 research and development budget?").json()
    assert r["abstained"]


def test_documents_endpoint_is_filtered_by_role():
    emp = {d["source"] for d in client.get("/documents", headers=EMP).json()}
    adm = {d["source"] for d in client.get("/documents", headers=ADMIN).json()}
    assert "hr_compensation_confidential.md" not in emp
    assert {"hr_compensation_confidential.md", "finance_budget_confidential.md"} <= adm


def test_ingest_is_admin_only():
    assert client.post("/ingest", headers=EMP).status_code == 403


def test_prompt_versions_listed_and_selectable():
    names = {p["name"] for p in client.get("/prompts", headers=EMP).json()}
    assert {"v1_baseline", "v2_cited", "v3_concise"} <= names
    assert ask(EMP, "How long is probation?", prompt_version="v3_concise").json()["prompt_version"] == "v3_concise"
    assert ask(EMP, "How long is probation?", prompt_version="does-not-exist").status_code == 400


def test_audit_log_written_without_secrets():
    from app import config

    text = config.AUDIT_LOG.read_text()
    assert '"event": "query"' in text and "demo-employee-key" not in text
