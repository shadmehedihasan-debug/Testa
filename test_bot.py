"""
Quick test suite: checks each bot against tricky questions.
Run:  python test_bot.py [tfidf|semantic|llm]

Each case is (question, expected FAQ question or None).  None = the bot should say it doesn't know.
Add your own cases - especially ones your FAQs do NOT cover.
"""
import sys

CASES = [
    # exact / easy
    ("How do I track my order?", "How can I track my order?"),
    ("What is your return policy?", "What is your return policy?"),
    # paraphrases (semantic bots should handle these; TF-IDF may not)
    ("Where is my package?", "How can I track my order?"),
    ("I want my money back", "How do I get a refund?"),
    ("I forgot my login", "How do I reset my password?"),
    ("Can you deliver abroad?", "Do you ship internationally?"),
    ("Do you take PayPal?", "What payment methods do you accept?"),
    # multi-part questions (matched FAQs are joined with " | ")
    ("How long is shipping and how do I get a refund?",
     "How long does shipping take? | How do I get a refund?"),
    ("Do you take PayPal and can I contact support?",
     "What payment methods do you accept? | How can I contact customer support?"),
    # near-misses that could trigger the wrong FAQ
    ("Can I return a gift card?", None),
    # out of scope - must NOT be answered
    ("What's the weather today?", None),
    ("Do you sell iPhones?", None),
    ("Who won the football match?", None),
    ("Can I get a discount code?", None),
]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "semantic"
    if mode == "tfidf":
        from chatbot import FAQChatbot
        bot = FAQChatbot()
    elif mode == "semantic":
        from ai_chatbot import SemanticFAQBot
        bot = SemanticFAQBot()
    else:
        from ai_chatbot import LLMFAQBot
        bot = LLMFAQBot()

    passed = 0
    for question, expected in CASES:
        answer, matched, score = bot.get_response(question)
        ok = matched == expected
        passed += ok
        print(f"{'PASS' if ok else 'FAIL'} | {question!r}")
        if not ok:
            print(f"       expected: {expected}\n       got:      {matched} (score {score:.2f})")
    print(f"\n{mode}: {passed}/{len(CASES)} passed")
    if mode == "llm":
        print("Note: for llm mode also read the answers manually - matching the right FAQ "
              "doesn't prove the wording is faithful.")


if __name__ == "__main__":
    main()
