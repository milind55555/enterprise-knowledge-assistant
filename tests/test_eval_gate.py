from eval import metrics, run_eval


def test_faithfulness_flags_invented_numbers():
    ctx = "Employees get 24 days of paid annual leave."
    assert metrics.faithfulness("Employees get 24 days of annual leave. [a.md]", ctx) == 1.0
    assert metrics.faithfulness("Employees get 30 days of annual leave.", ctx) == 0.0


def test_leak_detection():
    assert metrics.leaked(["hr.md"], "I don't have enough information in the knowledge base to answer that.", ["hr.md"], [])
    assert metrics.leaked(["a.md"], "Band is 14 to 22 LPA", [], ["14 to 22"])
    assert not metrics.leaked(["a.md"], "I don't have enough information in the knowledge base to answer that.", ["hr.md"], ["14 to 22"])


def test_eval_dataset_size_and_gate_passes():
    assert 30 <= len(run_eval.load_dataset()) <= 50
    assert run_eval.main([]) == 0
