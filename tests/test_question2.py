from __future__ import annotations

from pathlib import Path

import pytest

import question2.agent as q2_agent
from eval.reference import TICKET_EXPECTATIONS
from question2.agent import SupportPipeline
from shared.kb import article_ids
from shared.llm import _wants_verbose
from shared.tokens import count_tokens

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]
ALL_TICKETS = PUBLIC + ["T-1091"]
TARGET_PER_TICKET = 2500


def test_help_center_exceeds_context():  # slow (builds a 210k-token blob); do not change
    from shared.kb import expanded_help_center_blob
    assert count_tokens(expanded_help_center_blob()) > 200_000


def test_pipeline_resolves_public_tickets_and_holdout():
    pipe = SupportPipeline()
    for tid in ALL_TICKETS:
        exp = TICKET_EXPECTATIONS[tid]
        got = pipe.handle_ticket(tid)
        assert (got.action, got.kb_article) == (exp.action, exp.kb_article)


def test_cost_is_under_target_per_ticket():
    for tid in PUBLIC:
        pipe = SupportPipeline()
        pipe.handle_ticket(tid)
        assert pipe.session.total_points() <= TARGET_PER_TICKET, tid


def test_single_llm_call_on_happy_path():
    pipe = SupportPipeline()
    pipe.handle_ticket("T-1001")
    assert [c.label for c in pipe.session.calls] == ["compose"]


@pytest.mark.parametrize("tid", ALL_TICKETS)
def test_retrieval_recall(tid):
    pipe = SupportPipeline()
    case = pipe._triage(tid)
    assert TICKET_EXPECTATIONS[tid].kb_article in pipe._retrieve(case)


def test_prefix_is_cacheable():
    assert count_tokens(q2_agent.CACHED_PREFIX) >= 1024 + 50
    pipe = SupportPipeline()
    pipe.handle_ticket("T-1004")
    assert pipe.session.calls[0].cached_prefix_tokens > 1024


@pytest.mark.parametrize("tid", ALL_TICKETS)
def test_prompt_avoids_verbose_triggers(tid):
    pipe = SupportPipeline()
    case = pipe._triage(tid)
    pipe._compose(case, pipe._retrieve(case))
    assert pipe.session.calls[0].output_tokens < 100


def test_index_matches_real_articles():
    for aid in article_ids():
        assert aid in q2_agent._INDEX


def test_audit_rejects_invalid_output():
    pipe = SupportPipeline()
    assert pipe._audit("not json") is None
    assert pipe._audit('{"action":"REFUND","kb_article":"made-up-article"}') is None
    assert pipe._audit('{"action":"GIVE_COUPON","kb_article":"returns-within-30-days"}') is None
    ok = pipe._audit('{"action":"refund","kb_article":"returns-within-30-days"}')
    assert (ok.action, ok.kb_article) == ("REFUND", "returns-within-30-days")


def test_agent_source_is_clean():
    src = Path(q2_agent.__file__).read_text()
    for banned in ("eval.reference", "decide_order", "expanded_help_center_blob"):
        assert banned not in src
