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
EXCEL_FILE = r"C:\Users\phani\Downloads\PDT Analysis- Conley - Main.xlsx"
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
def load_all_sheets() -> dict[str, pd.DataFrame]:
    return pd.read_excel(EXCEL_FILE, sheet_name=None)  # all sheets

try:
    sheets = load_all_sheets()           # {sheet_name: DataFrame}
    df     = list(sheets.values())[0]   # primary sheet for metrics/charts default
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
            "Execute pandas code across any sheet. "
            "`sheets` is a dict of {sheet_name: DataFrame}. "
            "Individual sheets are also available as variables named by their sheet name (spaces replaced with _). "
            "Assign output to `result`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Pandas code. `sheets`, `pd`, `np` are available. "
                        "Access a sheet via `sheets['Sheet1']` or its variable name. "
                        "Must set `result`. E.g.: `result = sheets['Sales'].groupby('Region')['Revenue'].sum()`"
                    ),
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "create_chart",
        "description": "Create an interactive Plotly chart from any sheet.",
        "input_schema": {
            "type": "object",
            "properties": {
                "sheet_name": {"type": "string", "description": "Sheet name to chart from. Defaults to the first sheet."},
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
    local_ns: dict = {"sheets": sheets, "pd": pd, "np": np}
    # expose each sheet as a sanitised variable name
    for name, sdf in sheets.items():
        var = name.replace(" ", "_").replace("-", "_")
        local_ns[var] = sdf
    local_ns["df"] = df  # primary sheet shortcut
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
    chart_type  = params.get("chart_type", "bar")
    sheet_name  = params.get("sheet_name")
    x, y        = params.get("x"), params.get("y")
    color       = params.get("color")
    title       = params.get("title", "Chart")
    agg_func    = params.get("agg_func", "none")
    try:
        plot_df = (sheets.get(sheet_name) if sheet_name and sheet_name in sheets else df).copy()
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
    parts = []
    for name, sdf in sheets.items():
        buf = io.StringIO()
        sdf.info(buf=buf)
        parts.append(
            f"--- Sheet: '{name}' ({sdf.shape[0]:,} rows × {sdf.shape[1]} cols) ---\n"
            f"Columns & dtypes:\n{buf.getvalue()}\n"
            f"First 3 rows:\n{sdf.head(3).to_string()}\n"
            f"Statistics:\n{sdf.describe(include='all').to_string()}"
        )
    return "\n\n".join(parts)

SYSTEM = (
    f"You are an expert data analyst. The Excel workbook has {len(sheets)} sheet(s): "
    f"{', '.join(repr(n) for n in sheets)}.\n\n"
    + data_summary()
    + "\n\nAccess sheets via `sheets['name']` in query_data code. "
    "Always use query_data for accurate numbers. "
    "Use create_chart when the user asks for a chart or a visual would add insight. "
    "Keep answers concise."
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

sheet_summary = " | ".join(f"{n}: {sdf.shape[0]:,}r×{sdf.shape[1]}c" for n, sdf in sheets.items())
st.markdown(f"## 📊 Data Intelligence &nbsp;<sub style='font-size:13px;color:#888'>{EXCEL_FILE.split(chr(92))[-1]} · {len(sheets)} sheet(s) · {sheet_summary}</sub>", unsafe_allow_html=True)

# Key metric cards
numeric_cols = df.select_dtypes("number").columns[:5].tolist()
if numeric_cols:
    metric_cols = st.columns(len(numeric_cols))
    for i, col in enumerate(numeric_cols):
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
