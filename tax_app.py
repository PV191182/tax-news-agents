"""
Tax Assistant — Streamlit Web Interface
Run: streamlit run tax_app.py
"""

from __future__ import annotations
import os
import sys
from decimal import Decimal

import streamlit as st
import anthropic

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tax_engine import FilingStatus, TaxReturnInput, compute_return, format_result  # noqa: E402

client = anthropic.Anthropic()

# ---------------------------------------------------------------------------
# Tool definition
# ---------------------------------------------------------------------------

TOOLS: list[anthropic.Tool] = [
    {
        "name": "calculate_federal_taxes",
        "description": (
            "Estimate 2025 US federal income taxes. "
            "Call once you have filing status plus the relevant income/payment figures. "
            "Returns a formatted breakdown including AGI, taxable income, total tax, "
            "refundable credits, and refund or balance due."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "filing_status": {
                    "type": "string",
                    "enum": [
                        "single",
                        "married_filing_jointly",
                        "married_filing_separately",
                        "head_of_household",
                    ],
                    "description": "Filing status for the return.",
                },
                "wages_w2": {"type": "number", "description": "W-2 Box 1 wages in USD."},
                "federal_tax_withheld": {"type": "number", "description": "W-2 Box 2 federal tax withheld in USD."},
                "interest_income": {"type": "number", "description": "1099-INT ordinary interest in USD."},
                "ordinary_dividends": {"type": "number", "description": "1099-DIV Box 1a ordinary dividends in USD."},
                "self_employment_net_profit": {"type": "number", "description": "Schedule C net SE profit in USD."},
                "estimated_tax_payments": {"type": "number", "description": "Total quarterly estimated payments in USD."},
                "num_qualifying_children": {"type": "integer", "description": "Qualifying children under 17."},
                "age": {"type": "integer", "description": "Primary filer's age at end of 2025."},
                "spouse_age": {"type": "integer", "description": "Spouse's age (MFJ only)."},
                "blind": {"type": "boolean", "description": "True if primary filer is legally blind."},
                "spouse_blind": {"type": "boolean", "description": "True if spouse is legally blind."},
                "use_standard_deduction": {"type": "boolean", "description": "True to use standard deduction."},
                "itemized_deductions": {"type": "number", "description": "Total itemized deductions in USD."},
            },
            "required": ["filing_status"],
        },
    }
]

_STATUS_MAP = {
    "single": FilingStatus.SINGLE,
    "married_filing_jointly": FilingStatus.MFJ,
    "married_filing_separately": FilingStatus.MFS,
    "head_of_household": FilingStatus.HOH,
}


def _execute_tool(name: str, tool_input: dict) -> str:
    if name != "calculate_federal_taxes":
        return f"Unknown tool: {name}"
    try:
        status = _STATUS_MAP.get(tool_input.get("filing_status", "").lower())
        if status is None:
            return f"Invalid filing_status '{tool_input.get('filing_status')}'."
        inp = TaxReturnInput(
            filing_status=status,
            wages_w2=Decimal(str(tool_input.get("wages_w2", 0))),
            federal_tax_withheld=Decimal(str(tool_input.get("federal_tax_withheld", 0))),
            interest_income=Decimal(str(tool_input.get("interest_income", 0))),
            ordinary_dividends=Decimal(str(tool_input.get("ordinary_dividends", 0))),
            self_employment_net_profit=Decimal(str(tool_input.get("self_employment_net_profit", 0))),
            estimated_tax_payments=Decimal(str(tool_input.get("estimated_tax_payments", 0))),
            num_qualifying_children=int(tool_input.get("num_qualifying_children", 0)),
            age=int(tool_input.get("age", 0)),
            spouse_age=int(tool_input.get("spouse_age", 0)),
            blind=bool(tool_input.get("blind", False)),
            spouse_blind=bool(tool_input.get("spouse_blind", False)),
            use_standard_deduction=bool(tool_input.get("use_standard_deduction", True)),
            itemized_deductions=Decimal(str(tool_input.get("itemized_deductions", 0))),
        )
        return format_result(compute_return(inp))
    except Exception as exc:
        return f"Calculation error: {exc}"


SYSTEM = """You are a friendly 2025 federal income tax assistant.
Your job is to help users estimate their federal taxes using the calculate_federal_taxes tool.

Workflow:
1. Greet the user and ask what they need help with.
2. Gather the required inputs conversationally — don't fire a wall of questions at once:
   - Filing status (required)
   - W-2 wages and federal tax withheld
   - Self-employment income (if any)
   - Interest / dividend income (if any)
   - Estimated tax payments made (if any)
   - Number of qualifying children under 17
   - Ages (if 65+ or blind, for the enhanced standard deduction)
3. Once you have enough information, call calculate_federal_taxes.
4. Explain the result in plain English: AGI, taxable income, total tax, credits, and the refund or balance due.
5. Offer to recalculate with different numbers if the user wants to explore scenarios.

Always include: "This is a federal estimate only — not tax advice. Consult a CPA for your actual return."

Scope limitations (explain if asked):
- Capital gains are treated as ordinary income (no preferential rates)
- No AMT, NIIT, Additional Medicare Tax, or QBI deduction
- No state taxes, education credits, or dependent care credit
- Itemized deductions accepted as a lump sum only"""

# ---------------------------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="2025 Federal Tax Assistant", page_icon="🧾", layout="centered")
st.title("🧾 2025 Federal Tax Assistant")
st.caption("Powered by Claude · Federal estimates only · Not tax advice")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Render conversation history
for msg in st.session_state.messages:
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

# Chat input
if prompt := st.chat_input("Tell me about your tax situation..."):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        full_response = ""

        while True:
            with client.messages.stream(
                model="claude-opus-4-7",
                max_tokens=4096,
                thinking={"type": "adaptive"},
                system=SYSTEM,
                tools=TOOLS,
                messages=st.session_state.messages,
            ) as stream:
                for text in stream.text_stream:
                    full_response += text
                    placeholder.markdown(full_response + "▌")
                response = stream.get_final_message()

            if full_response:
                placeholder.markdown(full_response)

            st.session_state.messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break

            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    with st.spinner(f"Calculating taxes ({block.input.get('filing_status')})..."):
                        result = _execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result,
                    })

            st.session_state.messages.append({"role": "user", "content": tool_results})
