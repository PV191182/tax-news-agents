"""
Excel Data Agent — Streamlit
Upload an Excel file, ask questions, and generate charts powered by Claude.
Run: streamlit run excel_agent.py
"""

from __future__ import annotations
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
import anthropic

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "name": "query_data",
        "description": (
            "Execute pandas code to query, filter, aggregate, or compute statistics on the dataframe. "
            "The dataframe is available as `df`. Assign your output to a variable named `result`."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": (
                        "Python/pandas code. `df` is the dataframe, `pd` and `np` are available. "
                        "Must assign output to `result`. "
                        "Example: `result = df.groupby('Region')['Sales'].sum().reset_index()`"
                    ),
                }
            },
            "required": ["code"],
        },
    },
    {
        "name": "create_chart",
        "description": "Create an interactive Plotly chart from the dataframe columns.",
        "input_schema": {
            "type": "object",
            "properties": {
                "chart_type": {
                    "type": "string",
                    "enum": ["bar", "line", "scatter", "pie", "histogram", "box", "area"],
                    "description": "Chart type.",
                },
                "x": {"type": "string", "description": "Column for x-axis (or labels for pie chart)."},
                "y": {"type": "string", "description": "Column for y-axis (or values for pie chart)."},
                "color": {"type": "string", "description": "Column for color grouping (optional)."},
                "title": {"type": "string", "description": "Chart title."},
                "agg_func": {
                    "type": "string",
                    "enum": ["sum", "mean", "count", "max", "min", "none"],
                    "description": "Aggregation to apply before charting. Use 'none' for raw data.",
                },
            },
            "required": ["chart_type", "title"],
        },
    },
]

# ---------------------------------------------------------------------------
# Tool executors
# ---------------------------------------------------------------------------


def run_query(df: pd.DataFrame, code: str) -> str:
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


def run_chart(df: pd.DataFrame, params: dict):
    chart_type = params.get("chart_type", "bar")
    x = params.get("x")
    y = params.get("y")
    color = params.get("color")
    title = params.get("title", "Chart")
    agg_func = params.get("agg_func", "none")

    try:
        plot_df = df.copy()
        if agg_func != "none" and x and y:
            group_cols = [c for c in [x, color] if c]
            plot_df = plot_df.groupby(group_cols)[y].agg(agg_func).reset_index()

        kw: dict = {"title": title}
        if x:
            kw["x"] = x
        if y:
            kw["y"] = y
        if color:
            kw["color"] = color

        if chart_type == "bar":
            return px.bar(plot_df, **kw)
        elif chart_type == "line":
            return px.line(plot_df, **kw)
        elif chart_type == "scatter":
            return px.scatter(plot_df, **kw)
        elif chart_type == "pie":
            kw["names"] = kw.pop("x", None)
            kw["values"] = kw.pop("y", None)
            kw.pop("color", None)
            return px.pie(plot_df, **kw)
        elif chart_type == "histogram":
            kw.pop("y", None)
            return px.histogram(plot_df, **kw)
        elif chart_type == "box":
            return px.box(plot_df, **kw)
        elif chart_type == "area":
            return px.area(plot_df, **kw)
    except Exception as exc:
        return f"Chart error: {exc}"
    return None


# ---------------------------------------------------------------------------
# Data summary injected into system prompt
# ---------------------------------------------------------------------------


def data_summary(df: pd.DataFrame) -> str:
    buf = io.StringIO()
    df.info(buf=buf)
    return (
        f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns\n\n"
        f"Columns & dtypes:\n{buf.getvalue()}\n"
        f"First 5 rows:\n{df.head().to_string()}\n\n"
        f"Statistics:\n{df.describe(include='all').to_string()}"
    )


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def agent_turn(df: pd.DataFrame, messages: list) -> list:
    system = (
        "You are an expert data analyst. The user uploaded an Excel file with this data:\n\n"
        + data_summary(df)
        + "\n\nGuidelines:\n"
        "- Use query_data to get accurate numbers before stating facts.\n"
        "- Use create_chart whenever the user asks for a chart/graph/visualization, "
        "or when a visual would add clear insight.\n"
        "- Explain results in plain language after getting tool results."
    )

    full_text = ""
    placeholder = st.empty()

    while True:
        with client.messages.stream(
            model="claude-opus-4-7",
            max_tokens=4096,
            system=system,
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
                result_str = run_query(df, block.input["code"])
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": result_str}
                )

            elif block.name == "create_chart":
                fig = run_chart(df, block.input)
                if fig and not isinstance(fig, str):
                    st.session_state.charts[block.id] = fig
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": "Chart created successfully."}
                    )
                else:
                    error = fig if isinstance(fig, str) else "Unknown error."
                    tool_results.append(
                        {"type": "tool_result", "tool_use_id": block.id, "content": error}
                    )

        messages = messages + [{"role": "user", "content": tool_results}]

    placeholder.markdown(full_text)
    return messages


# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Excel Data Agent", page_icon="📊", layout="wide")
st.title("📊 Excel Data Agent")
st.caption("Upload an Excel file · Ask questions · Generate charts — Powered by Claude")

# Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Data")
    uploaded = st.file_uploader("Upload Excel file", type=["xlsx", "xls"])

    if uploaded:
        xf = pd.ExcelFile(uploaded)
        sheet = st.selectbox("Sheet", xf.sheet_names)
        data_key = f"{uploaded.name}_{uploaded.size}_{sheet}"

        if st.session_state.get("data_key") != data_key:
            df = pd.read_excel(uploaded, sheet_name=sheet)
            st.session_state.df = df
            st.session_state.data_key = data_key
            st.session_state.messages = []
            st.session_state.charts = {}

        st.success(f"{st.session_state.df.shape[0]:,} rows × {st.session_state.df.shape[1]} cols")
        st.dataframe(st.session_state.df.head(10), use_container_width=True)

# Guard ───────────────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    st.info("⬅ Upload an Excel file in the sidebar to get started.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "charts" not in st.session_state:
    st.session_state.charts = {}

df = st.session_state.df

# Conversation history ────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    role = msg["role"]
    content = msg["content"]

    if role == "user" and isinstance(content, str):
        with st.chat_message("user"):
            st.markdown(content)

    elif role == "assistant":
        texts = [b.text for b in content if hasattr(b, "type") and b.type == "text"]
        chart_ids = [
            b.id for b in content
            if hasattr(b, "type") and b.type == "tool_use" and b.name == "create_chart"
        ]
        if texts:
            with st.chat_message("assistant"):
                st.markdown("".join(texts))
                for cid in chart_ids:
                    if cid in st.session_state.charts:
                        st.plotly_chart(st.session_state.charts[cid], use_container_width=True)

# Chat input ──────────────────────────────────────────────────────────────────
if prompt := st.chat_input("Ask a question or request a chart..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        st.session_state.messages = agent_turn(df, list(st.session_state.messages))

        # Display charts produced in this turn
        last_asst = next(
            (m for m in reversed(st.session_state.messages) if m["role"] == "assistant"), None
        )
        if last_asst:
            for b in last_asst["content"]:
                if hasattr(b, "type") and b.type == "tool_use" and b.name == "create_chart":
                    if b.id in st.session_state.charts:
                        st.plotly_chart(st.session_state.charts[b.id], use_container_width=True)
