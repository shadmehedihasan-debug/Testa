"""Chat UI (Streamlit).  Run with:  streamlit run app.py"""
import os

import streamlit as st

from faq_data import SMALLTALK, log_feedback, log_interaction

st.set_page_config(page_title="FAQ Chatbot", page_icon="🤖")
st.title("🤖 FAQ Chatbot")

MODES = {
    "Basic (TF-IDF keywords)": "tfidf",
    "Semantic (embeddings)": "semantic",
    "AI (Claude + memory)": "llm",
    "AI: answer anything (Claude)": "llm_open",
}
# Streamlit Cloud: make the key from Settings -> Secrets available to the bots
try:
    if "ANTHROPIC_API_KEY" in st.secrets:
        os.environ["ANTHROPIC_API_KEY"] = st.secrets["ANTHROPIC_API_KEY"]
except Exception:
    pass

mode = MODES[st.sidebar.radio("Bot mode", list(MODES), index=3)]  # default: answer anything
st.sidebar.caption(
    {
        "tfidf": "Matches shared words. Fast, no downloads.",
        "semantic": "Matches meaning, e.g. 'money back' finds the refund FAQ.",
        "llm": "Finds the best FAQs, then Claude writes a natural reply and remembers the chat. Needs ANTHROPIC_API_KEY.",
        "llm_open": "Uses the FAQs when they apply and answers everything else from Claude's general knowledge, clearly labelled. Needs ANTHROPIC_API_KEY.",
    }[mode]
)
st.sidebar.caption("Chats and 👍/👎 ratings are saved to the logs/ folder. Run `python review_log.py` to see failures.")
if st.sidebar.button("Clear chat"):
    st.session_state.messages = []


@st.cache_resource
def load_bot(mode: str):
    if mode == "tfidf":
        from chatbot import FAQChatbot
        return FAQChatbot()
    from ai_chatbot import LLMFAQBot, SemanticFAQBot
    return SemanticFAQBot() if mode == "semantic" else LLMFAQBot(open_domain=(mode == "llm_open"))


try:
    with st.spinner("Loading bot..."):
        bot = load_bot(mode)
except Exception as e:  # missing package / API key
    st.error(f"Couldn't start this mode: {e}")
    st.info("To answer any question, add your key in the app's Settings -> Secrets:  "
            'ANTHROPIC_API_KEY = "your-key".  Meanwhile, pick "Basic" in the sidebar (FAQ answers only).')
    st.stop()

if not st.session_state.get("messages"):
    st.session_state.messages = [{"role": "assistant", "content": "Hi! How can I help you today?"}]


def rate(index: int, rating: str):
    msg = st.session_state.messages[index]
    log_feedback(msg["id"], rating)
    msg["rated"] = rating


for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("id"):  # feedback buttons on bot answers
            if m.get("rated"):
                st.caption("Thanks for your feedback! " + ("👍" if m["rated"] == "up" else "👎"))
            else:
                c1, c2, _ = st.columns([1, 1, 8])
                c1.button("👍", key=f"up{i}", on_click=rate, args=(i, "up"))
                c2.button("👎", key=f"down{i}", on_click=rate, args=(i, "down"))

if prompt := st.chat_input("Type your question..."):
    # conversation so far, without the greeting (used for follow-up questions in AI mode)
    history = [{"role": m["role"], "content": m.get("raw", m["content"])}
               for m in st.session_state.messages[1:]]
    st.session_state.messages.append({"role": "user", "content": prompt})

    try:
        answer, matched_q, score = bot.get_response(prompt, history) if mode != "tfidf" else bot.get_response(prompt)
    except Exception as e:
        answer, matched_q, score = f"Something went wrong: {e}", None, 0.0

    rid = log_interaction(mode, prompt, answer, matched_q, score)
    reply = answer
    if matched_q and matched_q != SMALLTALK and mode not in ("llm", "llm_open"):  # AI replies already end with their own Sources line
        reply += f"\n\n*Top FAQ: {matched_q} (similarity {score:.2f})*"
    st.session_state.messages.append({"role": "assistant", "content": reply, "raw": answer, "id": rid})
    st.rerun()
