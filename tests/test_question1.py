from __future__ import annotations

import json
from pathlib import Path

import pytest

import shared.tools as tools
from eval.reference import TICKET_EXPECTATIONS, decide_order
from question1 import agent as q1_agent
from question1.agent import SupportCopilot
from shared.llm import _wants_verbose
from shared.tokens import count_tokens

PUBLIC = ["T-1001", "T-1002", "T-1003", "T-1004"]
TARGET_PER_TICKET = 3000


def test_resolves_public_tickets():
    copilot = SupportCopilot()
    for tid in PUBLIC:
        expected = TICKET_EXPECTATIONS[tid]
        got = copilot.handle_ticket(tid)
        assert (got.action, got.kb_article) == (expected.action, expected.kb_article)


def test_holdout_subscription():
    got = SupportCopilot().handle_ticket("T-1091")
    exp = TICKET_EXPECTATIONS["T-1091"]
    assert (got.action, got.kb_article) == (exp.action, exp.kb_article)


def test_cost_is_under_target_per_ticket():
    for tid in PUBLIC:
        copilot = SupportCopilot()
        copilot.handle_ticket(tid)
        assert copilot.session.total_points() <= TARGET_PER_TICKET, tid


def test_one_llm_call_and_short_completion():
    copilot = SupportCopilot()
    copilot.handle_ticket("T-1001")
    assert len(copilot.session.calls) == 1
    assert copilot.session.calls[0].output_tokens < 100


@pytest.mark.parametrize("tid", PUBLIC + ["T-1091"])
def test_prompt_avoids_verbose_triggers_and_has_one_snapshot_marker(tid):
    messages = SupportCopilot().build_messages(tid)
    prompt = "\n\n".join(f"{m.role.upper()}:\n{m.content}" for m in messages)
    assert not _wants_verbose(prompt)
    assert prompt.count("ORDER_SNAPSHOT:") == 1


def test_prefix_is_cacheable_and_stable():
    assert count_tokens(q1_agent.CACHED_PREFIX) >= 1024 + 50  # safety margin over the threshold
    copilot = SupportCopilot()
    for tid in PUBLIC:
        copilot.handle_ticket(tid)
    assert all(c.cached_prefix_tokens > 1024 for c in copilot.session.calls)


def test_agent_does_not_touch_reference():
    src = Path(q1_agent.__file__).read_text()
    assert "eval.reference" not in src and "decide_order" not in src


# Edge cases: synthetic tickets, expected answers come from the reference (tests only).
BASE = dict(product_type="physical", delivered_days_ago=3, opened=False,
            license_revealed=False, customized=False, damage_reported=False)
EDGE = {
    "sub_12": dict(product_type="subscription", subscription_age_days=12),
    "sub_30": dict(product_type="subscription", subscription_age_days=30),
    "sub_31": dict(product_type="subscription", subscription_age_days=31),
    "unopened_d30": dict(delivered_days_ago=30),
    "unopened_d31": dict(delivered_days_ago=31),
    "opened_d20": dict(opened=True, delivered_days_ago=20),
    "damaged_opened_d7": dict(opened=True, damage_reported=True, delivered_days_ago=7),
    "damaged_opened_d8": dict(opened=True, damage_reported=True, delivered_days_ago=8),
    "damaged_unopened_d8": dict(damage_reported=True, delivered_days_ago=8),
    "custom_damaged_d1": dict(product_type="custom", customized=True, damage_reported=True, opened=True, delivered_days_ago=1),
    "digital_revealed": dict(product_type="digital", license_revealed=True, opened=True),
    "digital_not_revealed": dict(product_type="digital"),
}


@pytest.mark.parametrize("name", sorted(EDGE))
def test_edge_cases(name, tmp_path, monkeypatch):
    order = {**BASE, **EDGE[name], "order_id": "ord_x", "sku": "X", "title": "Test item"}
    (tmp_path / "orders").mkdir()
    (tmp_path / "tickets").mkdir()
    (tmp_path / "orders" / "ord_x.json").write_text(json.dumps(order))
    ticket = {"ticket_id": "T-X", "channel": "chat", "customer_id": "c", "order_id": "ord_x",
              "messages": [{"from": "customer", "text": "Please help with my order."}]}
    (tmp_path / "tickets" / "T-X.json").write_text(json.dumps(ticket))
    monkeypatch.setattr(tools, "DATA", tmp_path)
    exp = decide_order(order)
    got = SupportCopilot().handle_ticket("T-X")
    assert (got.action, got.kb_article) == (exp.action, exp.kb_article)
