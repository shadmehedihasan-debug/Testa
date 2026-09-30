"""
See where the bot fails.   Run:  python review_log.py
Shows (1) questions it couldn't answer and (2) answers users gave a thumbs-down.
Use these to add new FAQs / alternative wordings to faqs.json.
"""
import csv
from collections import Counter

from faq_data import CHAT_LOG, FEEDBACK_LOG


def read(path):
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


chats = read(CHAT_LOG)
feedback = {r["id"]: r["rating"] for r in read(FEEDBACK_LOG)}

if not chats:
    print("No chats logged yet. Talk to the bot first (streamlit run app.py).")
    raise SystemExit

unanswered = Counter(c["question"].strip().lower() for c in chats if c["answered"] == "no")
disliked = [c for c in chats if feedback.get(c["id"]) == "down"]
liked = sum(1 for c in chats if feedback.get(c["id"]) == "up")

print(f"Total chats: {len(chats)} | unanswered: {sum(unanswered.values())} | 👍 {liked} | 👎 {len(disliked)}\n")
print("== Unanswered questions (add FAQs or alternative wordings for these) ==")
for q, n in unanswered.most_common(20):
    print(f"  {n}x  {q}")
print("\n== Thumbs-down answers (check for wrong matches) ==")
for c in disliked[-20:]:
    print(f"  Q: {c['question']}\n     matched: {c['matched_faq'] or '-'} (score {c['score']}) | mode: {c['mode']}")
