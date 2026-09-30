"""
Shared helpers (no heavy dependencies):
  - load_faqs()        read faqs.json (edit that file to change the bot's knowledge)
  - split_questions()  break "X and Y?" into separate questions
  - answer_parts()     answer each part and combine the results
  - log_interaction()  / log_feedback()   save chats and thumbs up/down to logs/*.csv
"""
import csv
import json
import re
import uuid
from datetime import datetime
from pathlib import Path

BASE = Path(__file__).parent
FAQ_FILE = BASE / "faqs.json"
LOG_DIR = BASE / "logs"
CHAT_LOG = LOG_DIR / "chat_log.csv"
FEEDBACK_LOG = LOG_DIR / "feedback.csv"

FALLBACK = "Sorry, I couldn't find an answer to that. Try rephrasing, or contact support@example.com."


# ---------------------------------------------------------------- FAQs
def load_faqs(path=FAQ_FILE):
    with open(path, encoding="utf-8") as f:
        faqs = json.load(f)
    for i, faq in enumerate(faqs):
        if "q" not in faq or "a" not in faq:
            raise ValueError(f"FAQ #{i + 1} in {path} needs both a 'q' and an 'a'.")
        faq.setdefault("alt", [])
        faq.setdefault("exact", False)
    return faqs


def all_questions(faq):
    """The main question plus every alternative wording."""
    return [faq["q"]] + list(faq["alt"])


# ------------------------------------------------ multi-question support
def split_questions(text, max_parts=3):
    """'How long is shipping and how do I get a refund?' -> two questions.
    Only splits on 'and/also/plus' when both sides are 3+ words, so 'salt and pepper' stays whole."""
    text = text.strip()
    chunks = [c.strip() for c in re.split(r"\?+|;", text) if c.strip()]
    parts = []
    for chunk in chunks:
        sub = re.split(r"\s+(?:and|also|plus)\s+", chunk, flags=re.I)
        if len(sub) > 1 and all(len(s.split()) >= 3 for s in sub):
            parts.extend(s.strip() for s in sub)
        else:
            parts.append(chunk)
    return parts[:max_parts] or [text]


def answer_parts(text, single_fn):
    """single_fn(question) -> (answer, matched_q, score).  Answers each part and combines them."""
    parts = split_questions(text)
    if len(parts) == 1:
        return single_fn(parts[0])

    lines, matched, best, seen = [], [], 0.0, set()
    for part in parts:
        answer, matched_q, score = single_fn(part)
        label = part[0].upper() + part[1:] + "?"
        if matched_q is None:
            lines.append(f"**{label}**\nI couldn't find an answer to that one. Please contact support@example.com.")
        elif matched_q not in seen:  # don't repeat the same FAQ twice
            seen.add(matched_q)
            matched.append(matched_q)
            best = max(best, score)
            lines.append(f"**{label}**\n{answer}")
    if not matched:
        return FALLBACK, None, 0.0
    return "\n\n".join(lines), " | ".join(matched), best


# ------------------------------------------------------ logging / feedback
def _append(path, header, row):
    LOG_DIR.mkdir(exist_ok=True)
    is_new = not path.exists()
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if is_new:
            writer.writerow(header)
        writer.writerow(row)


def log_interaction(mode, question, answer, matched_q, score):
    """Save one exchange. Returns an id so feedback can be linked to it."""
    rid = uuid.uuid4().hex[:8]
    _append(
        CHAT_LOG,
        ["id", "time", "mode", "question", "answer", "matched_faq", "score", "answered"],
        [rid, datetime.now().isoformat(timespec="seconds"), mode, question, answer,
         matched_q or "", f"{score:.3f}", "yes" if matched_q else "no"],
    )
    return rid


def log_feedback(rid, rating):
    """rating is 'up' or 'down'."""
    _append(FEEDBACK_LOG, ["id", "time", "rating"],
            [rid, datetime.now().isoformat(timespec="seconds"), rating])
