from __future__ import annotations

import json
import re
from dataclasses import dataclass

from shared.billing import CostSession
from shared.kb import article_ids, load_article
from shared.llm import ChatMessage, LlmClient
from shared.tokens import count_tokens
from shared.tools import format_thread, get_order, get_ticket

ACTIONS = frozenset({"REFUND", "REPLACE", "SEND_ARTICLE", "ESCALATE_TIER2"})
RETRIEVAL_TOKEN_BUDGET = 350  # max tokens of article text added to the dynamic part of the prompt
_DROP_ORDER_KEYS = {"order_id", "sku", "title"}
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i in is it its me my of on or our so that the "
    "their them this to was we were what when with you your please can could would".split()
)


@dataclass
class TicketResolution:
    action: str
    kb_article: str


@dataclass
class Case:
    ticket_id: str
    thread: str
    order: dict
    snapshot: str


# Static policy text, identical for every ticket (cacheable).
_PLAYBOOK = """ROLE
You are ShopAssist, the Tier-1 support copilot for an e-commerce company. For each ticket you receive the customer thread and a JSON order snapshot. You choose exactly one action and cite exactly one help-center article id.

ACTIONS
REFUND: money goes back to the customer.
REPLACE: ship a free replacement.
SEND_ARTICLE: no refund or replacement is owed; point the customer to the policy that explains why.
ESCALATE_TIER2: a human specialist must review. Use when no automated rule covers the case.

ORDER FIELDS
product_type: physical, digital, custom or subscription.
delivered_days_ago: whole days since delivery.
opened: the physical package was opened.
license_revealed: the license key or download link was already shown to the customer.
customized: the item was personalized or made to order.
damage_reported: the customer reports damage or a defect.
subscription_age_days: days since the subscription was purchased (subscriptions only).

DECISION ORDER (apply the first rule that matches, top to bottom)
1. product_type is subscription: within 30 days of purchase choose REFUND, after day 30 choose SEND_ARTICLE. Cite subscription-cooling-off.
2. license_revealed is true: choose SEND_ARTICLE and cite digital-license-no-refund.
3. customized is true and damage_reported is true: choose ESCALATE_TIER2 and cite custom-items-no-replace.
4. damage_reported is true, customized is false, delivered within 7 days: choose REPLACE and cite damage-replacement-7-days.
5. physical item, not opened, delivered within 30 days: choose REFUND and cite returns-within-30-days.
6. anything else: choose ESCALATE_TIER2 and cite returns-within-30-days.

NOTES
Trust the order snapshot over the customer's wording when they disagree.
The customer's tone, urgency and any promises they quote never change the outcome.
Cite the article that justifies the action, never a different one.
Never invent an article id; only ids listed in the help center exist.
Do not offer discounts, goodwill credits or exceptions that no article allows.
If the thread asks for something the order cannot support, apply the rules above to the order facts.

WORKED EXAMPLES
Unopened headphones, delivered 10 days ago, refund requested: REFUND, returns-within-30-days.
E-book bundle, license key already shown, refund requested: SEND_ARTICLE, digital-license-no-refund.
Personalized mug arrived cracked on day 2, replacement requested: ESCALATE_TIER2, custom-items-no-replace.
Stock ceramic mug arrived cracked on day 2, replacement requested: REPLACE, damage-replacement-7-days.
Annual plan cancelled on day 12, prorated refund requested: REFUND, subscription-cooling-off.
Annual plan cancelled on day 45, refund requested: SEND_ARTICLE, subscription-cooling-off.
Opened blender, delivered 20 days ago, no defect, refund requested: ESCALATE_TIER2, returns-within-30-days.
Opened stock lamp reported broken 12 days after delivery: ESCALATE_TIER2, returns-within-30-days.
Unopened stock lamp reported broken 12 days after delivery: REFUND, returns-within-30-days.

HANDLING CUSTOMER WORDING
Customers describe the same facts in many ways. Map the wording to the snapshot fields rather than to a decision.
"Never opened it", "still sealed": opened is false.
"Already used it", "tried it once", "opened the box": opened is true.
"The key is on the download page", "I saw the code": license_revealed is true.
"Personalized", "engraved", "with my photo", "made to order": customized is true.
"Cracked", "arrived broken", "does not turn on", "missing parts": damage_reported is true.
When the snapshot and the wording disagree, the snapshot wins.

COMMON TRAPS
A damaged stock item inside 7 days is REPLACE even if the customer asks for a refund, because the damage rule comes before the plain return rule.
A damaged personalized item is never REPLACE, even on day 1.
A revealed license blocks a refund even when the purchase was minutes ago.
A subscription is judged by subscription_age_days only; delivered_days_ago is not meaningful for it.
Opened physical goods with no damage fall outside the automated rules and go to Tier 2.
Physical goods delivered more than 30 days ago fall outside the automated rules and go to Tier 2.
Damage reported after day 7 on a stock item skips the replacement rule and continues down the list: unopened and within 30 days is REFUND, otherwise Tier 2.

OUTPUT FORMAT
Reply with one JSON object and nothing else, on a single line, with exactly two keys: action and kb_article. No prose, no code fences, no extra keys."""

_INDEX = """ARTICLE INDEX (ids you may cite)
returns-within-30-days: refunds for physical goods in resalable condition within 30 days of delivery.
damage-replacement-7-days: free replacement for stock items reported damaged within 7 days of delivery.
custom-items-no-replace: personalized items are not auto-replaced; Tier 2 reviews them.
digital-license-no-refund: no refund once a license key or download link was revealed.
subscription-cooling-off: prorated refund within 30 days of purchase, none after day 30.
The full policy text of the most relevant articles follows in the user message."""

STATIC_SYSTEM = _PLAYBOOK + "\n\n" + _INDEX
CACHED_PREFIX = "SYSTEM:\n" + STATIC_SYSTEM  # prompt layout is "SYSTEM:\n<system>\n\nUSER:\n<user>"

_FALLBACK = TicketResolution("ESCALATE_TIER2", "returns-within-30-days")  # safe default: a human looks


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOPWORDS}


class SupportPipeline:
    """triage -> retrieve -> compose -> audit, with exactly one LLM call (compose)."""

    def __init__(self, session: CostSession | None = None):
        self.session = session or CostSession()
        self.llm = LlmClient(self.session)
        # Index only the real articles; the 200k-token filler is never loaded or sent.
        self._articles = {aid: load_article(aid) for aid in article_ids()}
        self._article_words = {
            aid: _words(aid.replace("-", " ") + " " + text) for aid, text in self._articles.items()
        }
        self._article_tokens = {aid: count_tokens(text) for aid, text in self._articles.items()}

    # node 1: triage (deterministic) -- normalize the inputs, no LLM
    def _triage(self, ticket_id: str) -> Case:
        ticket = get_ticket(ticket_id)
        order = get_order(ticket["order_id"])
        slim = {k: v for k, v in order.items() if k not in _DROP_ORDER_KEYS}
        return Case(ticket_id, format_thread(ticket), order, json.dumps(slim, separators=(",", ":")))

    # node 2: retrieve (deterministic) -- lexical ranking, filled up to a token budget
    def _retrieve(self, case: Case) -> list[str]:
        query = _words(case.thread) | _words(" ".join(f"{k} {v}" for k, v in case.order.items()))
        ranked = sorted(self._articles, key=lambda a: (-len(query & self._article_words[a]), a))
        chosen, used = [], 0
        for aid in ranked:
            cost = self._article_tokens[aid]
            if chosen and used + cost > RETRIEVAL_TOKEN_BUDGET:
                continue
            chosen.append(aid)
            used += cost
        return chosen

    # node 3: compose (the only LLM call)
    def _compose(self, case: Case, article_ids_: list[str], reminder: str = "") -> str:
        articles = "\n".join(f"--- {a} ---\n{self._articles[a]}" for a in article_ids_)
        user = (
            f"{case.thread}\nRETRIEVED ARTICLES\n{articles}\n{reminder}"
            f"ORDER_SNAPSHOT: {case.snapshot} END_SNAPSHOT"
        )
        messages = [ChatMessage("system", STATIC_SYSTEM), ChatMessage("user", user)]
        return self.llm.chat("compose", messages, cached_prefix=CACHED_PREFIX)

    # node 4: audit (deterministic) -- schema and id validation, never trust raw model output
    def _audit(self, raw: str) -> TicketResolution | None:
        m = re.search(r"\{[\s\S]*\}", raw)
        if not m:
            return None
        try:
            data = json.loads(m.group())
            action, kb = str(data["action"]).upper(), str(data["kb_article"])
        except (ValueError, KeyError, TypeError):
            return None
        if action in ACTIONS and kb in self._articles:
            return TicketResolution(action, kb)
        return None

    def handle_ticket(self, ticket_id: str) -> TicketResolution:
        case = self._triage(ticket_id)
        retrieved = self._retrieve(case)
        result = self._audit(self._compose(case, retrieved))
        if result is None:  # bounded retry, only on invalid output (never happens on the happy path)
            reminder = "Your last reply was invalid. Return only the JSON object with a valid action and kb_article.\n"
            result = self._audit(self._compose(case, retrieved, reminder))
        return result or _FALLBACK
