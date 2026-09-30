"""
Smarter FAQ bots
----------------
SemanticFAQBot : matches by MEANING using sentence embeddings (so "money back" ~ "refund")
LLMFAQBot      : retrieves the best FAQs semantically, then has Claude write a natural
                 reply from them, with conversation memory for follow-up questions.

Both handle multi-part questions ("How long is shipping and how do I get a refund?").

Setup for the LLM bot:
    pip install anthropic
    export ANTHROPIC_API_KEY="your-key"      (Windows: set ANTHROPIC_API_KEY=your-key)
"""

import os

import numpy as np

from faq_data import (FALLBACK, SMALLTALK, all_questions, answer_parts, load_faqs, log_interaction,
                      small_talk, split_questions)

FAQS = load_faqs()

EMBED_MODEL = "all-MiniLM-L6-v2"          # English, small, fast (~90 MB, downloads once)
# For other languages use: "paraphrase-multilingual-MiniLM-L12-v2"
LLM_MODEL = "claude-haiku-4-5-20251001"   # cheap + fast; use "claude-sonnet-5-5" for higher quality
CONFIDENCE_GATE = 0.35    # best FAQ below this -> skip the LLM entirely (no room to guess)
EXACT_MATCH_SCORE = 0.60  # "exact" FAQs are returned verbatim at/above this score


# --------------------------------------------------------------------------
# Semantic matching with sentence embeddings
# --------------------------------------------------------------------------
class SemanticFAQBot:
    def __init__(self, faqs=FAQS, threshold: float = 0.40):
        from sentence_transformers import SentenceTransformer

        self.faqs = faqs
        self.threshold = threshold
        self.model = SentenceTransformer(EMBED_MODEL)

        # Embed every question wording separately (plus question+answer), remembering which FAQ owns each.
        texts, owners = [], []
        for i, faq in enumerate(faqs):
            for wording in all_questions(faq):
                texts.append(wording)
                owners.append(i)
            texts.append(f"{faq['q']} {faq['a']}")
            owners.append(i)
        self.owners = np.array(owners)
        self.embeddings = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)

    def search(self, query: str, k: int = 3):
        """Top-k (faq, score) pairs, best first. Each FAQ's score = its best-matching wording."""
        q = self.model.encode(query, normalize_embeddings=True, convert_to_numpy=True)
        scores = self.embeddings @ q  # cosine similarity (vectors are normalized)
        per_faq = np.full(len(self.faqs), -1.0)
        np.maximum.at(per_faq, self.owners, scores)
        top = np.argsort(per_faq)[::-1][:k]
        return [(self.faqs[i], float(per_faq[i])) for i in top]

    def _single(self, question: str):
        faq, score = self.search(question, k=1)[0]
        if score < self.threshold:
            return FALLBACK, None, score
        return faq["a"], faq["q"], score

    def get_response(self, user_input: str, history=None):
        if not user_input.strip():
            return "Please type a question so I can help!", None, 0.0
        reply = small_talk(user_input)
        if reply:
            return reply, SMALLTALK, 1.0
        return answer_parts(user_input, self._single)


# --------------------------------------------------------------------------
# LLM-written answers (retrieval-augmented generation) + memory
# --------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a friendly customer-support assistant for an online store.
Answer the customer's question using ONLY the FAQ entries provided below.
- Write a natural, concise reply in your own words; combine entries if helpful.
- If the customer asks several questions, answer each one.
- Entries marked [EXACT] are policy text: copy their answer word for word, do not rephrase.
- Use the conversation history to understand follow-up questions.
- If the FAQs don't cover the question (or part of it), say so politely and suggest emailing support@example.com.
- Never invent policies, prices, or timeframes that aren't in the FAQs."""


GENERAL_PROMPT = """You are a helpful assistant in an online store's help chat.
The customer asked something our FAQ does not cover. Answer ANY question helpfully, accurately and concisely using general
knowledge. If the message is a greeting or casual chat, reply naturally and friendly, and offer to help.
- Never invent store-specific facts (policies, prices, stock, delivery times, account details). If the question depends on
  them, say you don't have that information and suggest emailing support@example.com.
- If you are not sure about something, say so instead of guessing."""

GENERAL_NOTE = "*General answer, not from our FAQ. Please double-check anything important.*"

SYSTEM_PROMPT_OPEN = SYSTEM_PROMPT.replace(
    "Answer the customer's question using ONLY the FAQ entries provided below.",
    "Answer the customer's question using the FAQ entries below wherever they apply.",
).replace(
    "- If the FAQs don't cover the question (or part of it), say so politely and suggest emailing support@example.com.",
    "- For any part the FAQs don't cover, answer from general knowledge on a separate line starting with "
    "'General answer (not from our FAQ):'. Never invent store-specific facts (policies, prices, timeframes); "
    "for those, say you don't have that information and suggest emailing support@example.com.",
)
assert SYSTEM_PROMPT_OPEN != SYSTEM_PROMPT


class LLMFAQBot:
    """open_domain=False: only answers from the FAQs.  open_domain=True: answers ANY question,
    using the FAQs where they apply and clearly labelled general knowledge otherwise."""

    def __init__(self, faqs=FAQS, retriever: "SemanticFAQBot | None" = None, top_k: int = 3,
                 gate: float = CONFIDENCE_GATE, open_domain: bool = False):
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError("Set the ANTHROPIC_API_KEY environment variable to use the LLM bot.")
        self.client = anthropic.Anthropic()
        self.retriever = retriever or SemanticFAQBot(faqs)
        self.top_k = top_k
        self.gate = gate
        self.open_domain = open_domain

    def _retrieval_query(self, user_input: str, history) -> str:
        # Short follow-ups ("what about international?") lack context, so add the last user turn.
        if history and len(user_input.split()) < 6:
            prev = [m["content"] for m in history if m["role"] == "user"]
            if prev:
                return f"{prev[-1]} {user_input}"
        return user_input

    def _retrieve(self, query: str):
        """Search each part of a multi-part question, then merge (keeping each FAQ's best score)."""
        parts = split_questions(query)
        k = self.top_k if len(parts) == 1 else 2
        merged = {}
        for part in parts:
            for faq, score in self.retriever.search(part, k=k):
                if faq["q"] not in merged or score > merged[faq["q"]][1]:
                    merged[faq["q"]] = (faq, score)
        return sorted(merged.values(), key=lambda h: h[1], reverse=True)

    def _general(self, user_input: str, history) -> str:
        """Answer outside the FAQ with general knowledge (open_domain mode only)."""
        messages = [{"role": m["role"], "content": m["content"]} for m in history[-8:]]
        messages.append({"role": "user", "content": user_input})
        resp = self.client.messages.create(
            model=LLM_MODEL, max_tokens=700, temperature=0, system=GENERAL_PROMPT, messages=messages
        )
        answer = "".join(b.text for b in resp.content if b.type == "text")
        return f"{answer}\n\n{GENERAL_NOTE}"

    def get_response(self, user_input: str, history=None):
        """history: list of {"role": "user"|"assistant", "content": str} from earlier turns."""
        if not user_input.strip():
            return "Please type a question so I can help!", None, 0.0
        history = history or []

        if not self.open_domain:  # in "answer anything" mode Claude handles greetings and chit-chat itself
            reply = small_talk(user_input)
            if reply:
                return reply, SMALLTALK, 1.0

        hits = self._retrieve(self._retrieval_query(user_input, history))
        best_faq, best_score = hits[0]

        # Safeguard 1: confidence gate - not similar enough to anything, so don't let the LLM guess.
        if best_score < self.gate:
            if not self.open_domain:
                return FALLBACK, None, best_score
            return self._general(user_input, history), None, best_score

        # Safeguard 2: a single high-stakes question is answered with the original text, no rewriting.
        if len(split_questions(user_input)) == 1 and best_faq.get("exact") and best_score >= EXACT_MATCH_SCORE:
            return best_faq["a"], best_faq["q"], best_score

        context = "\n\n".join(
            f"{'[EXACT] ' if f.get('exact') else ''}Q: {f['q']}\nA: {f['a']}" for f, _ in hits
        )
        messages = [{"role": m["role"], "content": m["content"]} for m in history[-8:]]
        messages.append({"role": "user", "content": user_input})

        resp = self.client.messages.create(
            model=LLM_MODEL,
            max_tokens=500,
            temperature=0,  # Safeguard 3: least creative = least likely to embellish
            system=f"{SYSTEM_PROMPT_OPEN if self.open_domain else SYSTEM_PROMPT}\n\nFAQ ENTRIES:\n{context}",
            messages=messages,
        )
        answer = "".join(b.text for b in resp.content if b.type == "text")

        # Safeguard 4: show which FAQs the answer was based on so users can verify it.
        sources = "; ".join(f"{f['q']} ({s:.2f})" for f, s in hits)
        return f"{answer}\n\n*Sources: {sources}*", best_faq["q"], best_score


# --------------------------------------------------------------------------
# Terminal demo:  python ai_chatbot.py [semantic|llm|llm_open]
# --------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    mode = sys.argv[1] if len(sys.argv) > 1 else "llm"
    bot = LLMFAQBot(open_domain=(mode == "llm_open")) if mode.startswith("llm") else SemanticFAQBot()
    history = []
    print(f"🤖 FAQ Bot ({mode} mode). Type 'quit' to exit.\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit", "bye"}:
            break
        answer, matched_q, score = bot.get_response(user, history)
        log_interaction(mode, user, answer, matched_q, score)
        print(f"🤖 {answer}\n   (top FAQ: '{matched_q}' | similarity {score:.2f})\n")
        history += [{"role": "user", "content": user}, {"role": "assistant", "content": answer}]
