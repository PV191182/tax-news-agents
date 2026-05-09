"""
Federal Tax Engine — Tax Year 2025
====================================
Deterministic federal tax calculator. Source: IRS Rev. Proc. 2024-40
as modified by the One Big Beautiful Bill Act (OBBBA, July 2025).

Scope (v1):
- Filing statuses: Single, MFJ, MFS, HoH
- Wage income (W-2), 1099-NEC self-employment, 1099-INT/DIV
- Standard deduction (with senior/blind add-ons and OBBBA bump)
- Federal income tax via 2025 brackets
- Self-employment tax (SE)
- Child Tax Credit (CTC) — simplified, including refundable ACTC
- Earned Income Tax Credit (EITC) — simplified
- Refund / balance-due reconciliation against withholding & estimated payments

NOT included in v1 (add later):
- Itemized deductions, AMT, NIIT, Additional Medicare Tax
- Capital gains preferential rates (treats all income as ordinary)
- State taxes, QBI deduction, education credits, dependent care credit
- Senior bonus deduction, tip/overtime exclusions from OBBBA

ALWAYS pair output with: "Estimate only. Not tax advice."
"""

from __future__ import annotations
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Constants — Tax Year 2025 (USD)
# ---------------------------------------------------------------------------

class FilingStatus(str, Enum):
    SINGLE = "single"
    MFJ = "married_filing_jointly"
    MFS = "married_filing_separately"
    HOH = "head_of_household"


# 2025 ordinary income brackets — Rev. Proc. 2024-40
# Each tuple: (upper_bound_of_bracket, marginal_rate). Top bracket uses inf.
INF = Decimal("Infinity")

BRACKETS_2025 = {
    FilingStatus.SINGLE: [
        (Decimal("11925"),  Decimal("0.10")),
        (Decimal("48475"),  Decimal("0.12")),
        (Decimal("103350"), Decimal("0.22")),
        (Decimal("197300"), Decimal("0.24")),
        (Decimal("250525"), Decimal("0.32")),
        (Decimal("626350"), Decimal("0.35")),
        (INF,               Decimal("0.37")),
    ],
    FilingStatus.MFJ: [
        (Decimal("23850"),  Decimal("0.10")),
        (Decimal("96950"),  Decimal("0.12")),
        (Decimal("206700"), Decimal("0.22")),
        (Decimal("394600"), Decimal("0.24")),
        (Decimal("501050"), Decimal("0.32")),
        (Decimal("751600"), Decimal("0.35")),
        (INF,               Decimal("0.37")),
    ],
    FilingStatus.MFS: [
        (Decimal("11925"),  Decimal("0.10")),
        (Decimal("48475"),  Decimal("0.12")),
        (Decimal("103350"), Decimal("0.22")),
        (Decimal("197300"), Decimal("0.24")),
        (Decimal("250525"), Decimal("0.32")),
        (Decimal("375800"), Decimal("0.35")),
        (INF,               Decimal("0.37")),
    ],
    FilingStatus.HOH: [
        (Decimal("17000"),  Decimal("0.10")),
        (Decimal("64850"),  Decimal("0.12")),
        (Decimal("103350"), Decimal("0.22")),
        (Decimal("197300"), Decimal("0.24")),
        (Decimal("250500"), Decimal("0.32")),
        (Decimal("626350"), Decimal("0.35")),
        (INF,               Decimal("0.37")),
    ],
}

# Standard deduction — 2025 with OBBBA bump
# (OBBBA: +$1,150 single/MFS/HoH-half, +$2,300 MFJ, on top of Rev. Proc. 2024-40)
STANDARD_DEDUCTION_2025 = {
    FilingStatus.SINGLE: Decimal("16150"),  # 15000 + 1150
    FilingStatus.MFJ:    Decimal("32300"),  # 30000 + 2300
    FilingStatus.MFS:    Decimal("16150"),
    FilingStatus.HOH:    Decimal("23850"),  # 22500 + 1350 approx; using announced figure
}

# Additional standard deduction (age 65+ or blind), per qualifying condition.
ADDL_STD_DEDUCTION_2025 = {
    FilingStatus.SINGLE: Decimal("2000"),
    FilingStatus.MFJ:    Decimal("1600"),
    FilingStatus.MFS:    Decimal("1600"),
    FilingStatus.HOH:    Decimal("2000"),
}

# FICA / Self-employment tax constants (2025)
SS_WAGE_BASE_2025 = Decimal("176100")
SS_RATE = Decimal("0.124")          # combined employer+employee for SE
MEDICARE_RATE = Decimal("0.029")    # combined for SE
SE_INCOME_FACTOR = Decimal("0.9235")  # 92.35% of net SE earnings is taxed
SE_DEDUCTION_FACTOR = Decimal("0.5")  # half of SE tax is above-the-line deduction

# Child Tax Credit (2025, OBBBA: max $2,200/child, refundable portion $1,700)
CTC_PER_CHILD = Decimal("2200")
CTC_REFUNDABLE_CAP = Decimal("1700")
CTC_PHASEOUT_THRESHOLD = {
    FilingStatus.SINGLE: Decimal("200000"),
    FilingStatus.MFJ:    Decimal("400000"),
    FilingStatus.MFS:    Decimal("200000"),
    FilingStatus.HOH:    Decimal("200000"),
}
CTC_PHASEOUT_RATE = Decimal("0.05")  # $50 per $1000 over threshold
CTC_EARNED_INCOME_THRESHOLD = Decimal("2500")
CTC_REFUNDABLE_RATE = Decimal("0.15")  # 15% of earned income over $2,500

# EITC — 2025 maximums (Rev. Proc. 2024-40). Simplified table.
# {num_children: (max_credit, earned_income_for_max, phaseout_begin_single, phaseout_begin_mfj, completed_phaseout_single, completed_phaseout_mfj)}
# Values from the IRS EITC tables for TY2025.
EITC_PARAMS_2025 = {
    0: {"max": Decimal("649"),  "earned_for_max": Decimal("8490"),  "phaseout_single": Decimal("10620"), "phaseout_mfj": Decimal("17730"), "complete_single": Decimal("19104"), "complete_mfj": Decimal("26214")},
    1: {"max": Decimal("4328"), "earned_for_max": Decimal("12730"), "phaseout_single": Decimal("23350"), "phaseout_mfj": Decimal("30470"), "complete_single": Decimal("50434"), "complete_mfj": Decimal("57554")},
    2: {"max": Decimal("7152"), "earned_for_max": Decimal("17880"), "phaseout_single": Decimal("23350"), "phaseout_mfj": Decimal("30470"), "complete_single": Decimal("57310"), "complete_mfj": Decimal("64430")},
    3: {"max": Decimal("8046"), "earned_for_max": Decimal("17880"), "phaseout_single": Decimal("23350"), "phaseout_mfj": Decimal("30470"), "complete_single": Decimal("61555"), "complete_mfj": Decimal("68675")},
}


# ---------------------------------------------------------------------------
# Input/Output dataclasses
# ---------------------------------------------------------------------------

@dataclass
class TaxReturnInput:
    """Everything the engine needs to compute a federal return.
    All money fields in USD as Decimal (or float / int — coerced)."""
    filing_status: FilingStatus
    # Personal
    age: int = 0                           # primary filer age at end of 2025
    spouse_age: int = 0                    # 0 if not MFJ
    blind: bool = False
    spouse_blind: bool = False
    num_qualifying_children: int = 0       # under 17, qualify for CTC
    num_other_dependents: int = 0          # ODC, $500 each (not yet implemented)

    # Income
    wages_w2: Decimal = Decimal("0")             # Box 1 of W-2
    federal_tax_withheld: Decimal = Decimal("0")  # Box 2 of W-2
    interest_income: Decimal = Decimal("0")      # 1099-INT box 1
    ordinary_dividends: Decimal = Decimal("0")   # 1099-DIV box 1a
    self_employment_net_profit: Decimal = Decimal("0")  # Schedule C net (1099-NEC etc.)

    # Payments
    estimated_tax_payments: Decimal = Decimal("0")

    # Adjustments / preferences
    use_standard_deduction: bool = True
    itemized_deductions: Decimal = Decimal("0")  # used if use_standard_deduction=False

    def __post_init__(self):
        # Coerce numeric fields to Decimal
        for fname in ("wages_w2", "federal_tax_withheld", "interest_income",
                      "ordinary_dividends", "self_employment_net_profit",
                      "estimated_tax_payments", "itemized_deductions"):
            v = getattr(self, fname)
            if not isinstance(v, Decimal):
                setattr(self, fname, Decimal(str(v)))


@dataclass
class TaxReturnResult:
    # Income build-up
    total_income: Decimal = Decimal("0")
    adjustments: Decimal = Decimal("0")          # e.g., 1/2 SE tax
    agi: Decimal = Decimal("0")
    deduction_taken: Decimal = Decimal("0")
    deduction_type: str = "standard"
    taxable_income: Decimal = Decimal("0")

    # Taxes
    income_tax_before_credits: Decimal = Decimal("0")
    se_tax: Decimal = Decimal("0")
    other_taxes: Decimal = Decimal("0")          # placeholder for AMT/NIIT later

    # Credits
    ctc_nonrefundable: Decimal = Decimal("0")
    ctc_refundable: Decimal = Decimal("0")
    eitc: Decimal = Decimal("0")

    # Final
    total_tax: Decimal = Decimal("0")
    total_payments: Decimal = Decimal("0")
    refund: Decimal = Decimal("0")               # positive = refund
    balance_due: Decimal = Decimal("0")          # positive = owe

    # Diagnostics
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            d[k] = float(v) if isinstance(v, Decimal) else v
        return d


# ---------------------------------------------------------------------------
# Core calculation helpers
# ---------------------------------------------------------------------------

def _money(x: Decimal) -> Decimal:
    """Round to cents."""
    return x.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)


def calculate_income_tax(taxable_income: Decimal, status: FilingStatus) -> Decimal:
    """Apply 2025 ordinary-income brackets."""
    if taxable_income <= 0:
        return Decimal("0")
    brackets = BRACKETS_2025[status]
    tax = Decimal("0")
    prev_cap = Decimal("0")
    for cap, rate in brackets:
        if taxable_income <= cap:
            tax += (taxable_income - prev_cap) * rate
            return _money(tax)
        else:
            tax += (cap - prev_cap) * rate
            prev_cap = cap
    return _money(tax)


def calculate_standard_deduction(inp: TaxReturnInput) -> Decimal:
    base = STANDARD_DEDUCTION_2025[inp.filing_status]
    addl = ADDL_STD_DEDUCTION_2025[inp.filing_status]
    extras = 0
    if inp.age >= 65: extras += 1
    if inp.blind:     extras += 1
    if inp.filing_status == FilingStatus.MFJ:
        if inp.spouse_age >= 65: extras += 1
        if inp.spouse_blind:     extras += 1
    return base + (addl * extras)


def calculate_se_tax(net_se_profit: Decimal) -> tuple[Decimal, Decimal]:
    """Returns (se_tax, deductible_half_of_se_tax)."""
    if net_se_profit <= 0:
        return Decimal("0"), Decimal("0")
    taxable_se = net_se_profit * SE_INCOME_FACTOR
    ss_taxable = min(taxable_se, SS_WAGE_BASE_2025)
    ss_part = ss_taxable * SS_RATE
    medicare_part = taxable_se * MEDICARE_RATE
    se_tax = _money(ss_part + medicare_part)
    deductible = _money(se_tax * SE_DEDUCTION_FACTOR)
    return se_tax, deductible


def calculate_ctc(agi: Decimal, earned_income: Decimal, num_kids: int,
                  income_tax_before_credits: Decimal,
                  status: FilingStatus) -> tuple[Decimal, Decimal]:
    """Returns (nonrefundable_CTC, refundable_ACTC)."""
    if num_kids <= 0:
        return Decimal("0"), Decimal("0")
    tentative = CTC_PER_CHILD * num_kids
    # Phase-out: $50 reduction per $1,000 (or fraction) AGI over threshold
    threshold = CTC_PHASEOUT_THRESHOLD[status]
    if agi > threshold:
        excess = agi - threshold
        # Round up to nearest $1,000
        increments = (excess / Decimal("1000")).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        if (excess % Decimal("1000")) > 0:
            increments += 1
        reduction = increments * Decimal("50")
        tentative = max(Decimal("0"), tentative - reduction)
    # Nonrefundable portion limited by tax liability before credits
    nonref = min(tentative, income_tax_before_credits)
    leftover = tentative - nonref
    # Refundable (ACTC): lesser of leftover, $1,700/child, or 15% of (earned - $2,500)
    refund_cap_per_kid = CTC_REFUNDABLE_CAP * num_kids
    earned_based = max(Decimal("0"), earned_income - CTC_EARNED_INCOME_THRESHOLD) * CTC_REFUNDABLE_RATE
    refundable = min(leftover, refund_cap_per_kid, earned_based)
    return _money(nonref), _money(refundable)


def calculate_eitc(earned_income: Decimal, agi: Decimal, num_kids: int,
                   status: FilingStatus) -> Decimal:
    """Simplified EITC — uses formula approach, not the official IRS table.
    Real implementation should use IRS EITC table for exact-dollar matches."""
    kids_key = min(num_kids, 3)
    p = EITC_PARAMS_2025[kids_key]
    income_for_phaseout = max(earned_income, agi)
    is_joint = status == FilingStatus.MFJ
    phaseout_begin = p["phaseout_mfj"] if is_joint else p["phaseout_single"]
    complete = p["complete_mfj"] if is_joint else p["complete_single"]

    # Phase-in: credit grows from 0 to max as earned_income grows from 0 to earned_for_max
    if earned_income <= 0:
        return Decimal("0")
    if earned_income < p["earned_for_max"]:
        credit = (p["max"] * earned_income / p["earned_for_max"])
    else:
        credit = p["max"]
    # Phase-out
    if income_for_phaseout > phaseout_begin:
        if income_for_phaseout >= complete:
            return Decimal("0")
        phaseout_range = complete - phaseout_begin
        reduction = p["max"] * (income_for_phaseout - phaseout_begin) / phaseout_range
        credit = max(Decimal("0"), credit - reduction)
    return _money(credit)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------

def compute_return(inp: TaxReturnInput) -> TaxReturnResult:
    r = TaxReturnResult()

    # 1. Total income
    r.total_income = _money(
        inp.wages_w2 + inp.interest_income + inp.ordinary_dividends
        + max(Decimal("0"), inp.self_employment_net_profit)
    )

    # 2. SE tax + adjustment
    r.se_tax, half_se = calculate_se_tax(inp.self_employment_net_profit)
    r.adjustments = half_se

    # 3. AGI
    r.agi = _money(r.total_income - r.adjustments)

    # 4. Deduction
    if inp.use_standard_deduction:
        r.deduction_taken = calculate_standard_deduction(inp)
        r.deduction_type = "standard"
    else:
        r.deduction_taken = inp.itemized_deductions
        r.deduction_type = "itemized"

    # 5. Taxable income
    r.taxable_income = max(Decimal("0"), _money(r.agi - r.deduction_taken))

    # 6. Income tax before credits
    r.income_tax_before_credits = calculate_income_tax(r.taxable_income, inp.filing_status)

    # 7. Credits
    earned_income = inp.wages_w2 + max(Decimal("0"), inp.self_employment_net_profit)
    r.ctc_nonrefundable, r.ctc_refundable = calculate_ctc(
        r.agi, earned_income, inp.num_qualifying_children,
        r.income_tax_before_credits, inp.filing_status,
    )
    r.eitc = calculate_eitc(earned_income, r.agi, inp.num_qualifying_children, inp.filing_status)

    # 8. Total tax
    income_tax_after_nonref = max(Decimal("0"), r.income_tax_before_credits - r.ctc_nonrefundable)
    r.total_tax = _money(income_tax_after_nonref + r.se_tax + r.other_taxes)

    # 9. Payments (withholding + est. payments + refundable credits)
    r.total_payments = _money(
        inp.federal_tax_withheld + inp.estimated_tax_payments
        + r.ctc_refundable + r.eitc
    )

    # 10. Refund or balance due
    diff = r.total_payments - r.total_tax
    if diff >= 0:
        r.refund = _money(diff)
    else:
        r.balance_due = _money(-diff)

    # Notes / caveats
    r.notes.append("Estimate only. Not tax advice. Tax year 2025 federal only.")
    if inp.itemized_deductions > 0 and inp.use_standard_deduction:
        r.notes.append("You provided itemized deductions but standard was used. "
                       "Set use_standard_deduction=False to compare.")
    if inp.self_employment_net_profit > 0 and inp.estimated_tax_payments == 0:
        r.notes.append("Self-employment income detected with no estimated tax payments — "
                       "you may owe an underpayment penalty.")
    return r


# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

def format_result(r: TaxReturnResult) -> str:
    lines = [
        "=" * 50,
        "  FEDERAL TAX ESTIMATE — TAX YEAR 2025",
        "=" * 50,
        f"  Total income            ${r.total_income:>12,.2f}",
        f"  Adjustments (1/2 SE)    ${r.adjustments:>12,.2f}",
        f"  AGI                     ${r.agi:>12,.2f}",
        f"  {r.deduction_type.title()} deduction       ${r.deduction_taken:>12,.2f}",
        f"  Taxable income          ${r.taxable_income:>12,.2f}",
        "-" * 50,
        f"  Income tax (pre-credit) ${r.income_tax_before_credits:>12,.2f}",
        f"  CTC (nonrefundable)    -${r.ctc_nonrefundable:>12,.2f}",
        f"  SE tax                 +${r.se_tax:>12,.2f}",
        f"  TOTAL TAX               ${r.total_tax:>12,.2f}",
        "-" * 50,
        f"  Refundable CTC/ACTC     ${r.ctc_refundable:>12,.2f}",
        f"  EITC                    ${r.eitc:>12,.2f}",
        f"  Total payments          ${r.total_payments:>12,.2f}",
        "=" * 50,
    ]
    if r.refund > 0:
        lines.append(f"  REFUND                  ${r.refund:>12,.2f}")
    else:
        lines.append(f"  BALANCE DUE             ${r.balance_due:>12,.2f}")
    lines.append("=" * 50)
    for n in r.notes:
        lines.append(f"  ⚠  {n}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Quick demo
    demo = TaxReturnInput(
        filing_status=FilingStatus.MFJ,
        age=38, spouse_age=36,
        num_qualifying_children=2,
        wages_w2=Decimal("125000"),
        federal_tax_withheld=Decimal("14000"),
        interest_income=Decimal("450"),
    )
    print(format_result(compute_return(demo)))
