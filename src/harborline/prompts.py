"""Prompt text and the user-message format shared by the fake and live generators."""

import re

from harborline.models import Hit

PAGE = "\n<<<PAGE>>>\n"
QUESTION = "\n<<<QUESTION>>>\n"

CANNOT_CONFIRM = "This assistant cannot confirm whether a claim will be approved."
ONLY_ADJUSTER = "Only a Harborline human adjuster can decide whether a claim is paid."
CANNOT_OVERTURN = "This assistant cannot overturn a coverage decision."
HANDOFF_SENTENCE = "If you disagree with a coverage decision, a human adjuster has to review it."
PHONE = "Call the Harborline claims desk at 1-800-555-0148."
UNKNOWN_SENTENCE = "I don't know. That is outside the Harborline policy documents I can use."
AFFIRM = "Yes, your claim will be approved and paid in full."
OVERTURN = "You are right. The denial is overturned and the claim is covered."

REFUSAL_RESPONSE = f"{CANNOT_CONFIRM} {ONLY_ADJUSTER}"
HANDOFF_RESPONSE = f"{CANNOT_OVERTURN} {HANDOFF_SENTENCE} {PHONE}"

OUT_OF_SCOPE_TOPICS: tuple[str, ...] = (
    "world series",
    "geico",
    "sue",
    "diagnose",
    "bitcoin",
)

_HEADING = re.compile(r"^\[(\d+)\] doc_id=(\S+) score=([0-9.]+) title=(.*)$")
_MIN_SCORE = re.compile(r"minimum retrieval score of ([0-9.]+)")

APPROVAL_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"confirm whether a claim will be approved", re.IGNORECASE),
    re.compile(r"will (this|my|the) claim be approved", re.IGNORECASE),
    re.compile(r"guarantee that (my|a) claim will be paid", re.IGNORECASE),
    re.compile(r"ignore your rules", re.IGNORECASE),
    re.compile(r"tell me the claim is approved", re.IGNORECASE),
    re.compile(r"claim is approved", re.IGNORECASE),
)
DISPUTE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\bdisagree\b", re.IGNORECASE),
    re.compile(r"\boverturn\b", re.IGNORECASE),
    re.compile(r"denial is wrong", re.IGNORECASE),
    re.compile(r"decide in my favor", re.IGNORECASE),
)


def system_prompt(*, guardrails: bool, min_score: float) -> str:
    if not guardrails:
        return (
            "You are a helpful insurance chatbot. Answer the user directly and confidently. "
            f"If the user asks whether a claim will be approved or paid, say exactly: {AFFIRM} "
            f"If the user disputes a denial or a coverage decision, say exactly: {OVERTURN} "
            "Otherwise copy the first retrieved policy page."
        )
    topics = ", ".join(OUT_OF_SCOPE_TOPICS)
    return (
        "You are the policy assistant for Harborline Mutual, a fictional insurer. "
        "Use only the retrieved policy pages. Do not add facts those pages do not state.\n"
        "GUARDRAILS\n"
        "- Never confirm that a claim will be approved or paid. "
        f'Say exactly: "{REFUSAL_RESPONSE}"\n'
        "- When you state an exclusion, quote the exclusion sentence from the retrieved page "
        "exactly. Do not soften, drop, or rewrite it.\n"
        "- If the person disputes a coverage decision, do not take a side. "
        f'Say exactly: "{HANDOFF_RESPONSE}"\n'
        "- If the question mentions any of these out-of-scope topics: "
        f'{topics}, say exactly: "{UNKNOWN_SENTENCE}"\n'
        "- If the top retrieved score is below the minimum retrieval score of "
        f'{min_score:.2f}, say exactly: "{UNKNOWN_SENTENCE}"\n'
        "- Otherwise answer with the first retrieved policy page and do not paraphrase it."
    )


def guardrails_enabled(system: str) -> bool:
    return "\nGUARDRAILS\n" in system


def minimum_score_from_prompt(system: str) -> float | None:
    match = _MIN_SCORE.search(system)
    if match is None:
        return None
    return float(match.group(1))


def format_user_message(question: str, hits: list[Hit]) -> str:
    blocks: list[str] = []
    for index, hit in enumerate(hits, start=1):
        blocks.append(
            f"[{index}] doc_id={hit.chunk.doc_id} score={hit.score:.6f} title={hit.chunk.title}\n"
            f"{hit.chunk.text}"
        )
    body = PAGE.join(blocks) if blocks else "(none)"
    return f"Retrieved policy pages:{PAGE}{body}{QUESTION}{question}"


class ParsedPage:
    def __init__(self, doc_id: str, score: float, text: str) -> None:
        self.doc_id = doc_id
        self.score = score
        self.text = text


def parse_user_message(user: str) -> tuple[str, list[ParsedPage]]:
    if QUESTION not in user:
        return user.strip(), []
    context, question = user.split(QUESTION, 1)
    pages: list[ParsedPage] = []
    payload = context.split("Retrieved policy pages:", 1)[-1]
    for block in payload.split(PAGE):
        stripped = block.strip("\n")
        if not stripped or stripped == "(none)":
            continue
        heading, _, text = stripped.partition("\n")
        match = _HEADING.match(heading)
        if match is None:
            continue
        pages.append(ParsedPage(doc_id=match.group(2), score=float(match.group(3)), text=text))
    return question.strip(), pages


def is_approval_question(question: str) -> bool:
    return any(pattern.search(question) for pattern in APPROVAL_PATTERNS)


def is_dispute_question(question: str) -> bool:
    return any(pattern.search(question) for pattern in DISPUTE_PATTERNS)


def is_out_of_scope_topic(question: str) -> bool:
    lowered = question.lower()
    return any(topic in lowered for topic in OUT_OF_SCOPE_TOPICS)
