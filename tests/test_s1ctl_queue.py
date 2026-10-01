"""Duplicate-submission guard: guards with a PENDING/RUNNING array task are not resubmitted."""
import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("s1ctl", Path(__file__).resolve().parents[1] / "s1ctl.py")
s1ctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(s1ctl)

SQUEUE = """954675_6|s1_zeroshot_24gb|RUNNING
954675_7|s1_zeroshot_24gb|PENDING
954675_8|s1_zeroshot_24gb|PENDING
954676_0|s1_zeroshot_40gb|PENDING
955117_1|s1_zeroshot_24gb|COMPLETING
123456|s1_tests|RUNNING
"""
ARRAYS = {"954675": ["a", "b", "c", "d", "e", "f", "nemotron", "granite", "stream"], "954676": ["lg4"], "955117": ["x", "y"]}


def test_parse_queued_guards_maps_task_index_to_guard():
    q = s1ctl.parse_queued_guards(SQUEUE.splitlines(), lambda base: ARRAYS.get(base))
    assert q == {"nemotron": ["954675_6"], "granite": ["954675_7"], "stream": ["954675_8"], "lg4": ["954676_0"]}


def test_parse_queued_guards_unexpanded_range_and_unknown_job():
    q = s1ctl.parse_queued_guards(["954675_[7-8%1]|s1_zeroshot_24gb|PENDING", "999_0|s1_zeroshot_24gb|PENDING"],
                                  lambda base: ARRAYS.get(base))
    assert set(q) == {"granite", "stream"} and "999_0" not in sum(q.values(), [])
