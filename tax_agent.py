"""
Tax Assistant Agent — 2025 Federal Tax Estimator
=================================================
Conversational agent powered by Claude claude-opus-4-7.
Wraps tax_engine.py via tool use; streams responses; maintains full
multi-turn conversation history including tool exchanges.

Install:  pip install anthropic
Run:      python tax_agent.py
"""

from __future__ import annotations
import json
import os
import sys
from decimal import Decimal

import anthropic

# tax_engine.py must be in the same directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tax_engine import FilingStatus, TaxReturnInput, compute_return, format_result  # noqa: E402

# ---------------------------------------------------------------------------
# Claude client
# ---------------------------------------------------------------------------

client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from environment

# ---------------------------------------------------------------------------
# Tool definition — input schema mirrors TaxReturnInput
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
                "wages_w2": {
                    "type": "number",
                    "description": "W-2 Box 1 wages in USD (default 0).",
                },
                "federal_tax_withheld": {
                    "type": "number",
                    "description": "W-2 Box 2 federal income tax withheld in USD (default 0).",
                },
                "interest_income": {
                    "type": "number",
                    "description": "1099-INT ordinary interest income in USD (default 0).",
                },
                "ordinary_dividends": {
                    "type": "number",
                    "description": "1099-DIV Box 1a ordinary dividends in USD (default 0).",
                },
                "self_employment_net_profit": {
                    "type": "number",
                    "description": "Schedule C / 1099-NEC net self-employment profit in USD (default 0).",
                },
                "estimated_tax_payments": {
                    "type": "number",
                    "description": "Total quarterly estimated tax payments made in USD (default 0).",
                },
                "num_qualifying_children": {
                    "type": "integer",
                    "description": "Qualifying children under 17 eligible for Child Tax Credit (default 0).",
                },
                "age": {
                    "type": "integer",
                    "description": "Primary filer's age at end of 2025 (affects standard deduction add-on, default 0).",
                },
                "spouse_age": {
                    "type": "integer",
                    "description": "Spouse's age at end of 2025 — only relevant for married_filing_jointly (default 0).",
                },
                "blind": {
                    "type": "boolean",
                    "description": "True if primary filer is legally blind (default false).",
                },
                "spouse_blind": {
                    "type": "boolean",
                    "description": "True if spouse is legally blind (default false).",
                },
                "use_standard_deduction": {
                    "type": "boolean",
                    "description": "True to use the standard deduction; false to use itemized_deductions (default true).",
                },
                "itemized_deductions": {
                    "type": "number",
                    "description": "Total itemized deductions in USD — only used when use_standard_deduction is false (default 0).",
                },
            },
            "required": ["filing_status"],
        },
    }
]

# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

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
            return (
                f"Invalid filing_status '{tool_input.get('filing_status')}'. "
                "Must be one of: single, married_filing_jointly, married_filing_separately, head_of_household."
            )

        inp = TaxReturnInput(
            filing_status=status,
            wages_w2=Decimal(str(tool_input.get("wages_w2", 0))),
            federal_tax_withheld=Decimal(str(tool_input.get("federal_tax_withheld", 0))),
            interest_income=Decimal(str(tool_input.get("interest_income", 0))),
            ordinary_dividends=Decimal(str(tool_input.get("ordinary_dividends", 0))),
            self_employment_net_profit=Decimal(
                str(tool_input.get("self_employment_net_profit", 0))
            ),
            estimated_tax_payments=Decimal(
                str(tool_input.get("estimated_tax_payments", 0))
            ),
            num_qualifying_children=int(tool_input.get("num_qualifying_children", 0)),
            age=int(tool_input.get("age", 0)),
            spouse_age=int(tool_input.get("spouse_age", 0)),
            blind=bool(tool_input.get("blind", False)),
            spouse_blind=bool(tool_input.get("spouse_blind", False)),
            use_standard_deduction=bool(tool_input.get("use_standard_deduction", True)),
            itemized_deductions=Decimal(str(tool_input.get("itemized_deductions", 0))),
        )

        return format_result(compute_return(inp))

    except Exception as exc:  # noqa: BLE001
        return f"Calculation error: {exc}"


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

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
- Itemized deductions accepted as a lump sum only (no individual item analysis)"""

# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------


def run() -> None:
    print("=" * 55)
    print("  2025 Federal Tax Assistant  (powered by Claude)")
    print("=" * 55)
    print("  Type 'quit' to exit.\n")

    messages: list[anthropic.MessageParam] = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Inner loop: handle tool-use rounds for this user turn
        while True:
            printed_prefix = False

            with client.messages.stream(
                model="claude-opus-4-7",
                max_tokens=4096,
                thinking={"type": "adaptive"},
                system=SYSTEM,
                tools=TOOLS,
                messages=messages,
            ) as stream:
                for text in stream.text_stream:
                    if not printed_prefix:
                        print("\nAssistant: ", end="", flush=True)
                        printed_prefix = True
                    print(text, end="", flush=True)

                response = stream.get_final_message()

            if printed_prefix:
                print()  # newline after streamed text

            # Preserve the full content (including thinking blocks) for history
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                break  # done with this user turn

            # Execute all tool calls and collect results
            tool_results: list[anthropic.ToolResultBlockParam] = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"\n  [Calculating... filing_status={block.input.get('filing_status')}]",
                          flush=True)
                    result = _execute_tool(block.name, block.input)
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result,
                        }
                    )

            messages.append({"role": "user", "content": tool_results})


if __name__ == "__main__":
    run()
