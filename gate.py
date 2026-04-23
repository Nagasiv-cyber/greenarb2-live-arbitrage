"""
gate.py — GreenArb 2.0 Sustainability & EAA Decision Gate
Evaluates whether a detected ADR arbitrage spread covers institutional costs,
and routes the decision to the appropriate hardware power mode.
"""

from typing import TypedDict

# ─── Institutional Constants ──────────────────────────────────────────────────
FEE_RATE                    = 0.0005   # 0.05% — institutional round-trip fee
ENERGY_COST_PER_TRADE_INR   = 0.50    # INR — baseline compute energy cost per scan
CO2_SAVED_PER_UNDERCLOCK_KG = 0.15   # kg CO₂ equivalent saved when hardware underclocks
EAA_EXECUTE_THRESHOLD       = 50.0    # INR — minimum Net EAA to trigger execution


class GateResult(TypedDict):
    status:       str    # "EXECUTE" | "REJECT - UNDERCLOCKING"
    power_mode:   str    # "1000W (Peak)" | "150W (Eco)"
    gross_profit: float  # INR
    total_fees:   float  # INR
    eaa:          float  # Net EAA in INR
    ccu_minted:   float  # Carbon Credit Units generated (0 on execute)
    fee_rate:     float
    trade_qty:    int


def evaluate_trade(
    gross_spread:  float,
    nse_price:     float,
    trade_qty:     int = 1_000,
) -> GateResult:
    """
    Run the Efficiency-Adjusted Alpha (EAA) gate for a given arbitrage spread.

    Parameters
    ----------
    gross_spread : float — Raw price difference (NYSE_INR - NSE_INR)
    nse_price    : float — NSE spot price in INR (used as notional base)
    trade_qty    : int   — Number of shares in the hypothetical trade block

    Returns
    -------
    GateResult dict with full decision breakdown.
    """

    # ── Step 1: Gross Profit ──────────────────────────────────────────────────
    gross_profit: float = gross_spread * trade_qty

    # ── Step 2: Transaction Fees (applied to NSE notional) ───────────────────
    notional_inr: float = nse_price * trade_qty
    total_fees:   float = notional_inr * FEE_RATE

    # ── Step 3: Net EAA ───────────────────────────────────────────────────────
    net_eaa: float = gross_profit - total_fees - ENERGY_COST_PER_TRADE_INR

    # ── Step 4: Decision Tree ─────────────────────────────────────────────────
    if net_eaa > EAA_EXECUTE_THRESHOLD:
        return GateResult(
            status       = "EXECUTE",
            power_mode   = "1000W (Peak)",
            gross_profit = round(gross_profit, 2),
            total_fees   = round(total_fees,   2),
            eaa          = round(net_eaa,      2),
            ccu_minted   = 0.0,
            fee_rate     = FEE_RATE,
            trade_qty    = trade_qty,
        )
    else:
        return GateResult(
            status       = "REJECT - UNDERCLOCKING",
            power_mode   = "150W (Eco)",
            gross_profit = round(gross_profit, 2),
            total_fees   = round(total_fees,   2),
            eaa          = round(net_eaa,      2),
            ccu_minted   = CO2_SAVED_PER_UNDERCLOCK_KG,
            fee_rate     = FEE_RATE,
            trade_qty    = trade_qty,
        )


if __name__ == "__main__":
    # Quick sanity check
    result = evaluate_trade(gross_spread=1.25, nse_price=1425.50, trade_qty=1000)
    for k, v in result.items():
        print(f"  {k:<14}: {v}")
