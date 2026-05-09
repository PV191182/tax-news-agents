"""
World News Agent — Streamlit Web App
Fetches the top 15 world headlines using Claude + live web search.
Run: streamlit run news_app.py
"""

from __future__ import annotations
import streamlit as st
import anthropic

client = anthropic.Anthropic()

TOOLS = [{"type": "web_search_20250305", "name": "web_search"}]

SYSTEM = """You are a world news assistant. Search the web for today's top stories.
Present exactly 15 headlines, numbered 1–15, in this format for each:

**Headline text** — Source, time
2-3 sentence summary of the story.

Cover a diverse mix: geopolitics, economy, science, technology, climate, and human interest.
After the list, tell the user they can ask follow-up questions about any story."""

INITIAL_PROMPT = "Search the web and show me the top 15 world news headlines right now."


def agent_turn(messages: list) -> list:
    """Run one agent turn with streaming, inside the current st.chat_message context."""
    full_text = ""
    placeholder = st.empty()

    while True:
        with client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=8096,
            system=SYSTEM,
            tools=TOOLS,
            messages=messages,
        ) as stream:
            for text in stream.text_stream:
                full_text += text
                placeholder.markdown(full_text + "▌")
            response = stream.get_final_message()

        messages = messages + [{"role": "assistant", "content": response.content}]

        if response.stop_reason != "tool_use":
            break

        tool_results = [
            {"type": "tool_result", "tool_use_id": b.id, "content": ""}
            for b in response.content
            if b.type == "tool_use"
        ]
        if tool_results:
            messages = messages + [{"role": "user", "content": tool_results}]

    placeholder.markdown(full_text)
    return messages


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="World Headlines", page_icon="🌍", layout="centered")

header_col, btn_col = st.columns([5, 1])
with header_col:
    st.title("🌍 World Headlines")
    st.caption("Top 15 live stories · Powered by Claude + web search")
with btn_col:
    st.write("")
    st.write("")
    refresh = st.button("🔄 Refresh")

# ── Session state ─────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Auto-fetch on first load or refresh ──────────────────────────────────────
if not st.session_state.messages or refresh:
    st.session_state.messages = [{"role": "user", "content": INITIAL_PROMPT}]
    with st.chat_message("assistant"):
        st.session_state.messages = agent_turn(st.session_state.messages)
    st.rerun()

# ── Display conversation history (skip the hidden auto-prompt at index 0) ─────
for i, msg in enumerate(st.session_state.messages):
    if i == 0:
        continue  # hide the auto "fetch headlines" prompt

    role = msg["role"]
    content = msg["content"]

    if role == "user" and isinstance(content, str):
        with st.chat_message("user"):
            st.markdown(content)
    elif role == "assistant":
        texts = [b.text for b in content if hasattr(b, "type") and b.type == "text"]
        if texts:
            with st.chat_message("assistant"):
                st.markdown("".join(texts))

# ── Follow-up chat ─────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask about any headline for more details..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        st.session_state.messages = agent_turn(st.session_state.messages)
