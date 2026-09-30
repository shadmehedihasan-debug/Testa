"""
FAQ Chatbot (basic mode)
------------------------
1. FAQs are loaded from faqs.json (questions, alternative wordings, answers)
2. Text is preprocessed with NLTK (lowercase, clean, tokenize, remove stopwords, lemmatize)
3. The user's question is matched to the most similar FAQ using TF-IDF + cosine similarity
4. The best matching answer is returned (or a fallback if nothing is similar enough)
5. Questions with several parts ("X and Y?") are answered part by part

Run in the terminal:  python chatbot.py
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from faq_data import FALLBACK, answer_parts, load_faqs, log_interaction

for pkg in ("punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"):
    nltk.download(pkg, quiet=True)  # one-time download; skipped if already installed

FAQS = load_faqs()  # edit faqs.json to change what the bot knows

_lemmatizer = WordNetLemmatizer()
_stop_words = set(stopwords.words("english"))


def preprocess(text: str) -> str:
    """Lowercase -> strip punctuation -> tokenize -> remove stopwords -> lemmatize."""
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    tokens = word_tokenize(text)
    return " ".join(_lemmatizer.lemmatize(t) for t in tokens if t not in _stop_words)


class FAQChatbot:
    def __init__(self, faqs=FAQS, threshold: float = 0.20):
        self.faqs = faqs
        self.threshold = threshold  # below this score we say "I don't know"
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2))
        # question (counted twice) + alternative wordings + answer text
        corpus = [preprocess(" ".join([f["q"], f["q"], *f["alt"], f["a"]])) for f in faqs]
        self.faq_vectors = self.vectorizer.fit_transform(corpus)

    def _single(self, question: str):
        cleaned = preprocess(question)
        if not cleaned.strip():
            return FALLBACK, None, 0.0
        scores = cosine_similarity(self.vectorizer.transform([cleaned]), self.faq_vectors).flatten()
        best = int(scores.argmax())
        if scores[best] < self.threshold:
            return FALLBACK, None, float(scores[best])
        return self.faqs[best]["a"], self.faqs[best]["q"], float(scores[best])

    def get_response(self, user_input: str, history=None):
        """Return (answer, matched_question, score). `history` is accepted but unused in this mode."""
        if not user_input.strip():
            return "Please type a question so I can help!", None, 0.0
        return answer_parts(user_input, self._single)


def main():
    bot = FAQChatbot()
    print("🤖 FAQ Bot: Hi! Ask me anything about orders, shipping, returns... (type 'quit' to exit)\n")
    while True:
        user = input("You: ").strip()
        if user.lower() in {"quit", "exit", "bye"}:
            print("🤖 FAQ Bot: Goodbye! 👋")
            break
        answer, matched_q, score = bot.get_response(user)
        log_interaction("tfidf", user, answer, matched_q, score)
        print(f"🤖 FAQ Bot: {answer}")
        if matched_q:
            print(f"   (matched: '{matched_q}' | similarity: {score:.2f})")
        print()


if __name__ == "__main__":
    main()
