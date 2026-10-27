"""Score-fusion decision rule and bootstrap helpers for the real-image evaluation."""
from qc.real.fusion import bootstrap, decide, metrics

RULES = {"bent_lead": {"verdict": "REWORK"}, "misplaced": {"verdict": "REWORK"},
         "cut_lead": {"verdict": "SCRAP"}, "damaged_case": {"verdict": "SCRAP"}, "unknown": {"verdict": "SCRAP"}}


def test_pass_when_neither_model_flags():
    assert decide(4.0, [{"cls": "misplaced", "conf": 0.1}], t_pc=5.0, t_seg=0.5, rules=RULES) == "PASS"


def test_seg_alone_can_flag_and_sets_type():
    assert decide(4.0, [{"cls": "misplaced", "conf": 0.6}], t_pc=5.0, t_seg=0.5, rules=RULES) == "REWORK"


def test_patchcore_flag_without_type_is_unknown_scrap():
    assert decide(6.0, [], t_pc=5.0, t_seg=0.5, rules=RULES) == "SCRAP"


def test_highest_confidence_detection_decides_type():
    dets = [{"cls": "cut_lead", "conf": 0.3}, {"cls": "bent_lead", "conf": 0.8}]
    assert decide(6.0, dets, t_pc=5.0, t_seg=0.5, rules=RULES) == "REWORK"


def test_metrics_and_bootstrap_bounds():
    yt = ["PASS", "PASS", "REWORK", "SCRAP"] * 5
    yp = ["PASS", "SCRAP", "REWORK", "SCRAP"] * 5
    m = metrics(yt, yp)
    assert m["accuracy"] == 0.75 and m["defect_recall"] == 1.0 and m["false_reject_rate"] == 0.5
    ci = bootstrap(yt, yp)
    assert ci["accuracy"][0] <= 0.75 <= ci["accuracy"][1]
