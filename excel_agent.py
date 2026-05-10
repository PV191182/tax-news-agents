"""
Excel Intelligence Agent — Streamlit
Chat with your data. Charts fill the page. Chat widget lives bottom-right.
Run: streamlit run excel_agent.py
"""

from __future__ import annotations
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import anthropic

# ── CONFIGURE YOUR FILE HERE ─────────────────────────────────────────────────
EXCEL_FILE = r"C:\Users\phani\Downloads\your_data.xlsx"   # ← set this path
SHEET_NAME = 0                                             # sheet index or name
# ─────────────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Page config & CSS
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Data Intelligence", page_icon="📊", layout="wide")

st.markdown("""
<style>
/* Hide Streamlit default padding */
.block-container { padding-top: 1rem; padding-bottom: 0rem; }

/* Chat widget panel */
.chat-panel {
    display: flex;
    flex-direction: column;
    height: 82vh;
    background: #ffffff;
    border-radius: 16px;
    border: 1px solid #e2e8f0;
    box-shadow: 0 8px 32px rgba(0,0,0,0.12);
    overflow: hidden;
}
.chat-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 14px 18px;
    font-weight: 600;
    font-size: 15px;
    border-radius: 16px 16px 0 0;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_excel(EXCEL_FILE, sheet_name=SHEET_NAME)

try:
    df = load_data()
except FileNotFoundError:
    st.error(f"File not found: `{EXCEL_FILE}`  \nUpdate the `EXCEL_FILE` path at the top of `excel_agent.py`.")
    st.stop()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "query_data",
        "description": (
            "Execute pandas code to query, filter, aggregate, or compute statistics. "
            "The dataframe is `df`. Assign output to `result`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Pandas code. `df`, `pd`, `np` are available. "
                        "Must set `result`. E.g.: `result = df.groupby('Region')['Sales'].sum()`"
                    ),
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "create_chart",
        "description": "Create an interactive Plotly chart from the dataframe.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "scatter", "pie", "histogram", "box", "area"],
                },
                "x":        {"type": "string", "description": "Column for x-axis / pie labels."},
                "y":        {"type": "string", "description": "Column for y-axis / pie values."},
                "color":    {"type": "string", "description": "Column for color grouping (optional)."},
                "title":    {"type": "string", "description": "Chart title."},
                "agg_func": {
                    "type": "string",
                    "enum": ["sum", "mean", "count", "max", "min", "none"],
                    "description": "Aggregation before charting.",
                },
            },
            "required": ["chart_type", "title"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------


def run_query(code: str) -> str:
    local_ns: dict = {"df": df.copy(), "pd": pd, "np": np}
    try:
        exec(code, local_ns)  # noqa: S102
        result = local_ns.get("result")
        if result is None:
            return "Code ran but `result` was never assigned."
        if isinstance(result, pd.DataFrame):
            return result.to_string(max_rows=50)
        if isinstance(result, pd.Series):
            return result.to_string()
        return str(result)
    except Exception as exc:
        return f"Error: {exc}"


def run_chart(params: dict):
    chart_type = params.get("chart_type", "bar")
    x, y      = params.get("x"), params.get("y")
    color     = params.get("color")
    title     = params.get("title", "Chart")
    agg_func  = params.get("agg_func", "none")
    try:
        plot_df = df.copy()
        if agg_func != "none" and x and y:
            group_cols = [c for c in [x, color] if c]
            plot_df = plot_df.groupby(group_cols)[y].agg(agg_func).reset_index()
        kw: dict = {"title": title}
        if x:     kw["x"]     = x
        if y:     kw["y"]     = y
        if color: kw["color"] = color
        if chart_type == "bar":       return px.bar(plot_df, **kw)
        if chart_type == "line":      return px.line(plot_df, **kw)
        if chart_type == "scatter":   return px.scatter(plot_df, **kw)
        if chart_type == "pie":
            kw["names"]  = kw.pop("x", None)
            kw["values"] = kw.pop("y", None)
            kw.pop("color", None)
            return px.pie(plot_df, **kw)
        if chart_type == "histogram": kw.pop("y", None); return px.histogram(plot_df, **kw)
        if chart_type == "box":       return px.box(plot_df, **kw)
        if chart_type == "area":      return px.area(plot_df, **kw)
    except Exception as exc:
        return f"Chart error: {exc}"


# ---------------------------------------------------------------------------
# Data summary for system prompt
# ---------------------------------------------------------------------------

def data_summary() -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return (
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n"
        f"Columns & dtypes:\n{buf.getvalue()}\n"
        f"First 5 rows:\n{df.head().to_string()}\n\n"
        f"Statistics:\n{df.describe(include='all').to_string()}"
    )

SYSTEM = (
    "You are an expert data analyst. The user has an Excel dataset:\n\n"
    + data_summary()
    + "\n\nAlways use query_data for accurate numbers. Use create_chart when the user "
    "asks for a chart or when a visualization adds insight. Keep answers concise."
)

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

def agent_turn(messages: list, placeholder) -> list:
    full_text = ""

    while True:
        with client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=4096,
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

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if block.name == "query_data":
                res = run_query(block.input["code"])
                tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": res})
            elif block.name == "create_chart":
                fig = run_chart(block.input)
                if fig and not isinstance(fig, str):
                    st.session_state.charts[block.id] = fig
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": "Chart created."})
                else:
                    err = fig if isinstance(fig, str) else "Chart failed."
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": err})

        messages = messages + [{"role": "user", "content": tool_results}]

    placeholder.markdown(full_text)
    return messages


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "charts" not in st.session_state:
    st.session_state.charts = {}   # tool_use_id → figure

# ---------------------------------------------------------------------------
# Layout: title row
# ---------------------------------------------------------------------------

st.markdown(f"## 📊 Data Intelligence &nbsp;<sub style='font-size:13px;color:#888'>{EXCEL_FILE.split('\\')[-1]} · {df.shape[0]:,} rows × {df.shape[1]} cols</sub>", unsafe_allow_html=True)

# Key metric cards
metric_cols = st.columns(min(len(df.select_dtypes("number").columns), 5))
for i, col in enumerate(df.select_dtypes("number").columns[:5]):
    with metric_cols[i]:
        st.metric(col, f"{df[col].sum():,.0f}", f"avg {df[col].mean():,.1f}")

st.divider()

# ---------------------------------------------------------------------------
# Two-column body: charts left | chat right
# ---------------------------------------------------------------------------

col_charts, col_chat = st.columns([3, 1], gap="medium")

# ── LEFT: charts area ────────────────────────────────────────────────────────
with col_charts:
    if not st.session_state.charts:
        st.markdown("""
        <div style="display:flex;align-items:center;justify-content:center;
                    height:60vh;color:#aaa;flex-direction:column;gap:12px">
            <span style="font-size:48px">📈</span>
            <span style="font-size:16px">Charts will appear here as you explore your data</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        charts = list(st.session_state.charts.values())
        if len(charts) == 1:
            st.plotly_chart(charts[0], use_container_width=True)
        else:
            for i in range(0, len(charts), 2):
                c1, c2 = st.columns(2)
                with c1:
                    st.plotly_chart(charts[i], use_container_width=True)
                with c2:
                    if i + 1 < len(charts):
                        st.plotly_chart(charts[i + 1], use_container_width=True)

# ── RIGHT: chat widget ───────────────────────────────────────────────────────
with col_chat:
    st.markdown('<div class="chat-header">💬 Ask Your Data</div>', unsafe_allow_html=True)

    # Scrollable message area
    with st.container(height=560, border=True):
        if not st.session_state.messages:
            st.markdown(
                "<div style='color:#aaa;font-size:13px;padding:8px'>Ask me anything about your data — "
                "totals, trends, comparisons, or say \"show me a chart of...\"</div>",
                unsafe_allow_html=True,
            )
        for msg in st.session_state.messages:
            role    = msg["role"]
            content = msg["content"]
            if role == "user" and isinstance(content, str):
                with st.chat_message("user"):
                    st.markdown(content)
            elif role == "assistant":
                texts = [b.text for b in content if hasattr(b, "type") and b.type == "text"]
                if texts:
                    with st.chat_message("assistant"):
                        st.markdown("".join(texts))

    # Input form — Enter to submit
    with st.form("chat_form", clear_on_submit=True, enter_to_submit=True, border=False):
        inp_col, btn_col = st.columns([5, 1])
        with inp_col:
            user_input = st.text_input(
                "", placeholder="Ask anything about your data...",
                label_visibility="collapsed",
            )
        with btn_col:
            submitted = st.form_submit_button("➤", type="primary", use_container_width=True)

    if submitted and user_input.strip():
        st.session_state.messages.append({"role": "user", "content": user_input.strip()})

        # Stream response in a temporary placeholder inside the chat column
        with st.container(border=True):
            with st.chat_message("assistant"):
                placeholder = st.empty()
                st.session_state.messages = agent_turn(
                    list(st.session_state.messages), placeholder
                )

        st.rerun()
