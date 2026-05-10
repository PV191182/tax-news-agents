"""
DataChat — Customer-facing Data Intelligence Platform
Run: streamlit run excel_agent.py
"""

from __future__ import annotations
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import anthropic

# ── CONFIG ────────────────────────────────────────────────────────────────────
EXCEL_FILE  = r"C:\Users\phani\Downloads\PDT Analysis- Conley - Main.xlsx"
APP_NAME    = "DataChat"
APP_TAGLINE = "Explore your data through conversation"
# ─────────────────────────────────────────────────────────────────────────────

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title=APP_NAME,
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { font-family: 'Inter', -apple-system, sans-serif !important; box-sizing: border-box; }

/* Hide Streamlit chrome */
#MainMenu, footer, header, .stDeployButton { visibility: hidden; display: none; }

/* Full bleed */
.block-container { padding: 0 !important; max-width: 100% !important; }

/* Page background */
.stApp { background: #f0f4f8; }

/* Metric delta — hide the redundant delta row */
[data-testid="stMetricDelta"] { display: none; }

/* Chat message bubbles */
[data-testid="stChatMessage"] {
    background: #f8fafc;
    border-radius: 10px;
    padding: 10px 14px;
    margin: 6px 0;
    border: 1px solid #e2e8f0;
}

/* Send button gradient */
[data-testid="stFormSubmitButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #2563eb, #7c3aed) !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    color: white !important;
}

/* Secondary buttons — chip style */
button[kind="secondary"] {
    border-radius: 20px !important;
    font-size: 12px !important;
    border-color: #e2e8f0 !important;
    color: #475569 !important;
    background: white !important;
}
button[kind="secondary"]:hover {
    border-color: #2563eb !important;
    color: #2563eb !important;
    background: #eff6ff !important;
}

/* Form — remove default border */
[data-testid="stForm"] { border: none !important; padding: 0 !important; box-shadow: none !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; height: 4px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #cbd5e1; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #94a3b8; }

/* Plotly chart — remove extra padding */
[data-testid="stPlotlyChart"] { border-radius: 12px; overflow: hidden; }

/* Text input */
[data-testid="stTextInput"] input {
    border-radius: 8px !important;
    border-color: #e2e8f0 !important;
    font-size: 13px !important;
}
[data-testid="stTextInput"] input:focus {
    border-color: #2563eb !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
}

/* Container borders */
[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 12px !important;
    border-color: #e2e8f0 !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load data — all sheets
# ---------------------------------------------------------------------------

@st.cache_data
def load_all_sheets() -> dict[str, pd.DataFrame]:
    return pd.read_excel(EXCEL_FILE, sheet_name=None)

try:
    sheets = load_all_sheets()
    df     = list(sheets.values())[0]
except FileNotFoundError:
    st.error(f"File not found: `{EXCEL_FILE}`. Update the EXCEL_FILE path at the top of `excel_agent.py`.")
    st.stop()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "query_data",
        "description": (
            "Execute pandas code across any sheet. "
            "`sheets` is a dict {sheet_name: DataFrame}. "
            "Each sheet is also accessible as a variable (spaces → _). "
            "Assign output to `result`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Pandas code. `sheets`, `pd`, `np` available. "
                        "Must assign to `result`. "
                        "E.g.: `result = sheets['Sales'].groupby('Region')['Rev'].sum()`"
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
                "sheet_name": {"type": "string", "description": "Sheet to chart from (defaults to first sheet)."},
                "chart_type": {"type": "string", "enum": ["bar", "line", "scatter", "pie", "histogram", "box", "area"]},
                "x":        {"type": "string", "description": "Column for x-axis / pie labels."},
                "y":        {"type": "string", "description": "Column for y-axis / pie values."},
                "color":    {"type": "string", "description": "Column for color grouping (optional)."},
                "title":    {"type": "string", "description": "Chart title."},
                "agg_func": {"type": "string", "enum": ["sum", "mean", "count", "max", "min", "none"]},
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
    for name, sdf in sheets.items():
        local_ns[name.replace(" ", "_").replace("-", "_")] = sdf
    local_ns["df"] = df
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
    sname      = params.get("sheet_name")
    x, y       = params.get("x"), params.get("y")
    color      = params.get("color")
    title      = params.get("title", "Chart")
    agg_func   = params.get("agg_func", "none")
    try:
        plot_df = (sheets.get(sname) if sname and sname in sheets else df).copy()
        if agg_func != "none" and x and y:
            group_cols = [c for c in [x, color] if c]
            plot_df = plot_df.groupby(group_cols)[y].agg(agg_func).reset_index()
        kw: dict = {"title": title, "template": "plotly_white"}
        if x:     kw["x"]     = x
        if y:     kw["y"]     = y
        if color: kw["color"] = color
        fig = {
            "bar":       lambda: px.bar(plot_df, **kw),
            "line":      lambda: px.line(plot_df, **kw),
            "scatter":   lambda: px.scatter(plot_df, **kw),
            "histogram": lambda: px.histogram(plot_df, **{k: v for k, v in kw.items() if k != "y"}),
            "box":       lambda: px.box(plot_df, **kw),
            "area":      lambda: px.area(plot_df, **kw),
            "pie":       lambda: px.pie(plot_df, names=x, values=y, title=title, template="plotly_white"),
        }.get(chart_type, lambda: None)()
        if fig:
            fig.update_layout(
                font_family="Inter",
                title_font_size=15,
                title_font_color="#1e293b",
                margin=dict(t=48, l=16, r=16, b=16),
                plot_bgcolor="white",
                paper_bgcolor="white",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
        return fig
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
            f"{buf.getvalue()}\n"
            f"First 3 rows:\n{sdf.head(3).to_string()}\n"
            f"Statistics:\n{sdf.describe(include='all').to_string()}"
        )
    return "\n\n".join(parts)

SYSTEM = (
    f"You are a friendly, expert data analyst assistant embedded in a customer-facing dashboard. "
    f"The workbook has {len(sheets)} sheet(s): {', '.join(repr(n) for n in sheets)}.\n\n"
    + data_summary()
    + "\n\nGuidelines:\n"
    "- Always use query_data to get accurate numbers before stating facts.\n"
    "- Use create_chart for any chart/graph/visualization request.\n"
    "- Write in a clear, professional tone — the audience is business users, not engineers.\n"
    "- Be concise. Lead with the key insight, then supporting detail."
)

# ---------------------------------------------------------------------------
# Smart suggestions from column names
# ---------------------------------------------------------------------------

def get_suggestions() -> list[tuple[str, str]]:
    cols = df.columns.tolist()
    num  = df.select_dtypes("number").columns.tolist()
    cat  = df.select_dtypes(["object", "category"]).columns.tolist()
    sugg = [("📊 Summarize", "Give me a high-level summary of this data")]
    if num:
        sugg.append((f"📈 Trend of {num[0]}", f"Show me a chart of {num[0]} over time or by category"))
    if cat and num:
        sugg.append((f"🔍 {num[0]} by {cat[0]}", f"Break down {num[0]} by {cat[0]} with a bar chart"))
    if len(num) >= 2:
        sugg.append((f"⚖️ Compare", f"Compare {num[0]} and {num[1]} across the data"))
    return sugg[:4]

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
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": "Chart created successfully."})
                else:
                    tool_results.append({"type": "tool_result", "tool_use_id": block.id, "content": str(fig)})
        messages = messages + [{"role": "user", "content": tool_results}]

    placeholder.markdown(full_text)
    return messages

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []
if "charts" not in st.session_state:
    st.session_state.charts = {}
if "pending_q" not in st.session_state:
    st.session_state.pending_q = None

# ---------------------------------------------------------------------------
# ── HEADER ──
# ---------------------------------------------------------------------------

file_label = EXCEL_FILE.split("\\")[-1]
st.markdown(f"""
<div style="
    background: linear-gradient(135deg, #1e3a5f 0%, #2563eb 60%, #7c3aed 100%);
    padding: 18px 32px 16px;
    display: flex; align-items: center; justify-content: space-between;
    box-shadow: 0 2px 12px rgba(0,0,0,0.18);
">
  <div>
    <span style="color:white;font-size:22px;font-weight:700;letter-spacing:-0.5px">💬 {APP_NAME}</span>
    <span style="color:rgba(255,255,255,0.65);font-size:13px;margin-left:12px">{APP_TAGLINE}</span>
  </div>
  <div style="
    background:rgba(255,255,255,0.15); border:1px solid rgba(255,255,255,0.25);
    border-radius:20px; padding:4px 14px;
    color:rgba(255,255,255,0.85); font-size:12px; font-weight:500;
  ">📄 {file_label}</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ── METRIC CARDS ──
# ---------------------------------------------------------------------------

numeric_cols = df.select_dtypes("number").columns[:5].tolist()
if numeric_cols:
    accent_colors = ["#2563eb", "#7c3aed", "#0891b2", "#059669", "#d97706"]
    cards_html = '<div style="display:flex;gap:12px;padding:16px 24px 0;">'
    for i, col in enumerate(numeric_cols):
        color = accent_colors[i % len(accent_colors)]
        total = df[col].sum()
        avg   = df[col].mean()
        cards_html += f"""
        <div style="flex:1;background:white;border-radius:12px;padding:16px 18px;
                    box-shadow:0 1px 4px rgba(0,0,0,0.07);border-top:3px solid {color};">
          <div style="color:#94a3b8;font-size:10px;font-weight:600;text-transform:uppercase;
                      letter-spacing:0.6px;margin-bottom:6px">{col}</div>
          <div style="color:#1e293b;font-size:20px;font-weight:700">{total:,.0f}</div>
          <div style="color:#94a3b8;font-size:11px;margin-top:3px">avg {avg:,.1f}</div>
        </div>"""
    st.markdown(cards_html + "</div>", unsafe_allow_html=True)

st.markdown('<div style="height:12px"></div>', unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# ── TWO-COLUMN BODY ──
# ---------------------------------------------------------------------------

col_charts, col_chat = st.columns([3, 1], gap="medium")

# ── LEFT: charts ─────────────────────────────────────────────────────────────
with col_charts:
    # Toolbar
    toolbar_l, toolbar_r = st.columns([6, 1])
    with toolbar_l:
        st.markdown(
            '<p style="color:#475569;font-size:13px;font-weight:600;margin:0;'
            'padding:0 8px">📈 Visualizations</p>',
            unsafe_allow_html=True,
        )
    with toolbar_r:
        if st.session_state.charts:
            if st.button("🗑 Clear", key="clear_charts", type="secondary", use_container_width=True):
                st.session_state.charts = {}
                st.rerun()

    if not st.session_state.charts:
        st.markdown("""
        <div style="
            background:white; border-radius:16px; min-height:65vh;
            display:flex; flex-direction:column; align-items:center;
            justify-content:center; gap:12px;
            border: 2px dashed #e2e8f0; margin:4px 0;
        ">
          <div style="font-size:52px">📊</div>
          <div style="color:#334155;font-size:16px;font-weight:600">Your charts will appear here</div>
          <div style="color:#94a3b8;font-size:13px">Ask the assistant to visualize your data</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        charts = list(st.session_state.charts.values())
        if len(charts) == 1:
            with st.container(border=False):
                st.plotly_chart(charts[0], use_container_width=True)
        else:
            for i in range(0, len(charts), 2):
                c1, c2 = st.columns(2, gap="small")
                with c1:
                    st.plotly_chart(charts[i], use_container_width=True)
                with c2:
                    if i + 1 < len(charts):
                        st.plotly_chart(charts[i + 1], use_container_width=True)

# ── RIGHT: chat ───────────────────────────────────────────────────────────────
with col_chat:
    # Chat header + clear button
    h1, h2 = st.columns([3, 1])
    with h1:
        st.markdown("""
        <div style="
            background:linear-gradient(135deg,#2563eb,#7c3aed);
            border-radius:12px 0 0 0; padding:12px 16px;
            color:white; font-size:14px; font-weight:600;
        ">🤖 Data Assistant</div>
        """, unsafe_allow_html=True)
    with h2:
        if st.button("Clear", key="clear_chat", type="secondary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Input at top
    with st.form("chat_form", clear_on_submit=True, enter_to_submit=True, border=False):
        inp_c, btn_c = st.columns([5, 1])
        with inp_c:
            user_input = st.text_input(
                "", placeholder="Ask anything about your data...",
                label_visibility="collapsed",
            )
        with btn_c:
            submitted = st.form_submit_button("➤", type="primary", use_container_width=True)

    # Suggestion chips — shown only when chat is empty
    if not st.session_state.messages and not st.session_state.pending_q:
        suggestions = get_suggestions()
        st.markdown('<div style="margin:6px 0 2px;"><span style="color:#94a3b8;font-size:11px;font-weight:500">TRY ASKING</span></div>', unsafe_allow_html=True)
        chip_cols = st.columns(len(suggestions))
        for i, (label, question) in enumerate(suggestions):
            with chip_cols[i]:
                if st.button(label, key=f"chip_{i}", use_container_width=True, type="secondary"):
                    st.session_state.pending_q = question
                    st.rerun()

    # Messages area
    with st.container(height=480, border=True):
        if not st.session_state.messages:
            st.markdown("""
            <div style="padding:20px 8px;text-align:center;">
              <div style="font-size:32px;margin-bottom:10px">👋</div>
              <div style="color:#334155;font-size:14px;font-weight:600;margin-bottom:6px">
                Hello! I'm your data assistant.
              </div>
              <div style="color:#94a3b8;font-size:12px;line-height:1.6">
                Ask me to summarize, filter, calculate,<br>or visualize your data.
              </div>
            </div>
            """, unsafe_allow_html=True)
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

    # Handle suggestion chip click
    question = st.session_state.pending_q
    if question:
        st.session_state.pending_q = None
        st.session_state.messages.append({"role": "user", "content": question})

    # Handle form submission
    elif submitted and user_input.strip():
        question = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": question})

    # Run agent if there's a new question
    if question:
        # Auto-clear after 5 assistant responses
        if sum(1 for m in st.session_state.messages if m["role"] == "assistant") >= 5:
            st.session_state.messages = [{"role": "user", "content": question}]

        with st.container(border=True):
            with st.chat_message("assistant"):
                with st.spinner("Analyzing your data..."):
                    placeholder = st.empty()
                st.session_state.messages = agent_turn(
                    list(st.session_state.messages), placeholder
                )
        st.rerun()
