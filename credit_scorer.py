from enum import Enum
from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from mpesa_parser import ParsedTransaction, TransactionType


class CreditRiskTier(str, Enum):
    EXCELLENT = "EXCELLENT"      # 750 - 850
    GOOD = "GOOD"                # 650 - 749
    FAIR = "FAIR"                # 550 - 649
    POOR = "POOR"                # 450 - 549
    VERY_POOR = "VERY_POOR"      # 300 - 449


class SpendingHabits(BaseModel):
    total_spend: float
    category_totals: Dict[str, float]
    category_percentages: Dict[str, float]
    essential_ratio: float = Field(..., description="Proportion of spending on Utilities, Groceries, Rent, Education, Healthcare")
    luxury_ratio: float = Field(..., description="Proportion of spending on Liquor, Nightlife, Bars, Lounges")
    gambling_ratio: float = Field(..., description="Proportion of spending on Betting, Casinos")
    debt_service_ratio: float = Field(..., description="Proportion of outflow dedicated to loan repayments")
    savings_ratio: float = Field(..., description="Proportion of outflow channeled to Saccos, MMFs, Investments")


class CashFlowMetrics(BaseModel):
    total_inflow: float
    total_outflow: float
    net_cashflow: float
    inflow_outflow_ratio: float
    average_tx_amount: float
    fuliza_count: int
    fuliza_total_amount: float
    negative_balance_count: int
    latest_balance: Optional[float] = None


class CreditScoreReport(BaseModel):
    credit_score: int = Field(..., ge=300, le=850)
    risk_tier: CreditRiskTier
    recommended_limit_kes: float
    confidence: float
    spending_habits: SpendingHabits
    cash_flow: CashFlowMetrics
    positive_factors: List[str]
    risk_factors: List[str]
    rule_flags: List[str]


class MpesaCreditScorer:
    """
    Evaluates borrower creditworthiness based on M-Pesa transaction history,
    categorized spending habits, cash flow dynamics, and Fuliza overdraft stress.
    """

    ESSENTIAL_CATEGORIES = {
        "Utilities",
        "Groceries",
        "Rent & Housing",
        "Education",
        "Healthcare",
    }

    def score_transactions(
        self,
        enriched_transactions: List[dict]
    ) -> CreditScoreReport:
        """
        Calculates credit score from a list of enriched transaction dicts:
        each item having {'parsed': ParsedTransaction, 'category': str, 'confidence': float}
        """
        if not enriched_transactions:
            # Baseline default for empty history
            return CreditScoreReport(
                credit_score=300,
                risk_tier=CreditRiskTier.VERY_POOR,
                recommended_limit_kes=0.0,
                confidence=0.0,
                spending_habits=SpendingHabits(
                    total_spend=0.0,
                    category_totals={},
                    category_percentages={},
                    essential_ratio=0.0,
                    luxury_ratio=0.0,
                    gambling_ratio=0.0,
                    debt_service_ratio=0.0,
                    savings_ratio=0.0,
                ),
                cash_flow=CashFlowMetrics(
                    total_inflow=0.0,
                    total_outflow=0.0,
                    net_cashflow=0.0,
                    inflow_outflow_ratio=0.0,
                    average_tx_amount=0.0,
                    fuliza_count=0,
                    fuliza_total_amount=0.0,
                    negative_balance_count=0,
                ),
                positive_factors=[],
                risk_factors=["No transaction history provided."],
                rule_flags=["EMPTY_HISTORY"],
            )

        # 1. Aggregate Cash Flow & Categories
        total_inflow = 0.0
        total_outflow = 0.0
        fuliza_count = 0
        fuliza_total = 0.0
        negative_balance_count = 0
        latest_balance = None
        category_totals: Dict[str, float] = {}
        all_amounts: List[float] = []

        for item in enriched_transactions:
            tx: ParsedTransaction = item["parsed"]
            cat: str = item["category"]

            all_amounts.append(tx.amount)

            if tx.balance is not None:
                latest_balance = tx.balance
                if tx.balance < 0:
                    negative_balance_count += 1

            if tx.fuliza_amount and tx.fuliza_amount > 0:
                fuliza_count += 1
                fuliza_total += tx.fuliza_amount

            # Inflow vs Outflow logic
            if tx.tx_type == TransactionType.RECEIVE_MONEY:
                total_inflow += tx.amount
            else:
                total_outflow += tx.amount
                category_totals[cat] = category_totals.get(cat, 0.0) + tx.amount

        # Calculate category percentages against total outflow
        cat_pcts: Dict[str, float] = {}
        outflow_denom = total_outflow if total_outflow > 0 else 1.0
        for c, amt in category_totals.items():
            cat_pcts[c] = round((amt / outflow_denom) * 100, 2)

        # Key ratios
        essential_spend = sum(category_totals.get(c, 0.0) for c in self.ESSENTIAL_CATEGORIES)
        essential_ratio = round(essential_spend / outflow_denom, 4)

        luxury_spend = category_totals.get("Luxury", 0.0)
        luxury_ratio = round(luxury_spend / outflow_denom, 4)

        gambling_spend = category_totals.get("Gambling", 0.0)
        gambling_ratio = round(gambling_spend / outflow_denom, 4)

        debt_service_spend = category_totals.get("Loan Repayment", 0.0)
        debt_service_ratio = round(debt_service_spend / outflow_denom, 4)

        savings_spend = category_totals.get("Savings & Investments", 0.0)
        savings_ratio = round(savings_spend / outflow_denom, 4)

        net_cashflow = round(total_inflow - total_outflow, 2)
        inflow_outflow_ratio = round(total_inflow / outflow_denom, 4)
        avg_tx = round(sum(all_amounts) / len(all_amounts), 2) if all_amounts else 0.0

        # 2. Score Computation (Baseline 600)
        score = 600
        positive_factors: List[str] = []
        risk_factors: List[str] = []
        rule_flags: List[str] = []

        # (a) Cash Flow & Inflow Health
        if total_inflow > total_outflow:
            margin = (total_inflow - total_outflow) / (total_inflow if total_inflow > 0 else 1)
            pts = min(int(margin * 60), 60)
            score += pts
            positive_factors.append(f"Net positive cash flow with {round(margin * 100, 1)}% savings margin (+{pts} pts)")
        else:
            deficit_pts = min(int((total_outflow - total_inflow) / outflow_denom * 50), 60)
            score -= deficit_pts
            risk_factors.append(f"Outflows exceed inflows (-{deficit_pts} pts)")
            rule_flags.append("CASHFLOW_DEFICIT")

        # (b) Buffer & Balance
        if latest_balance is not None:
            if latest_balance >= 5000:
                score += 25
                positive_factors.append("Healthy terminal M-Pesa balance buffer (>KES 5,000) (+25 pts)")
            elif latest_balance < 0:
                score -= 35
                risk_factors.append("Terminal M-Pesa balance is in deficit/negative (-35 pts)")
                rule_flags.append("NEGATIVE_TERMINAL_BALANCE")
            elif latest_balance < 500:
                score -= 15
                risk_factors.append("Low terminal balance liquidity (<KES 500) (-15 pts)")

        # (c) Fuliza Overdraft & Distress
        if fuliza_count == 0:
            score += 30
            positive_factors.append("Zero Fuliza overdraft dependency recorded (+30 pts)")
        else:
            fuliza_penalty = min(fuliza_count * 25 + int(fuliza_total * 0.02), 120)
            score -= fuliza_penalty
            risk_factors.append(f"Active Fuliza overdraft reliance: {fuliza_count} instances totaling KES {fuliza_total:,.2f} (-{fuliza_penalty} pts)")
            rule_flags.append("FULIZA_OVERDRAFT_USER")

        if negative_balance_count > 0:
            neg_penalty = min(negative_balance_count * 20, 60)
            score -= neg_penalty
            risk_factors.append(f"{negative_balance_count} negative balance events recorded (-{neg_penalty} pts)")
            rule_flags.append("OVERDRAFT_INCIDENCE")

        # (d) Gambling / Betting Activity (Severe Negative Signal)
        if gambling_ratio > 0:
            if gambling_ratio > 0.10:
                score -= 140
                risk_factors.append(f"High gambling spend detected: {round(gambling_ratio * 100, 1)}% of total outflows (-140 pts)")
                rule_flags.append("HIGH_GAMBLING_RISK")
            else:
                score -= 60
                risk_factors.append(f"Gambling / sports betting expenditure detected: {round(gambling_ratio * 100, 1)}% of outflows (-60 pts)")
                rule_flags.append("GAMBLING_ACTIVITY")
        else:
            score += 15
            positive_factors.append("Zero gambling/betting activity detected (+15 pts)")

        # (e) Luxury & Liquor Stores (Discretionary Consumption)
        if luxury_ratio > 0.30:
            score -= 50
            risk_factors.append(f"Excessive luxury / liquor store spend: {round(luxury_ratio * 100, 1)}% of total outflows (-50 pts)")
            rule_flags.append("HIGH_LUXURY_CONSUMPTION")
        elif luxury_ratio > 0.15:
            score -= 25
            risk_factors.append(f"Elevated discretionary luxury / nightlife spend: {round(luxury_ratio * 100, 1)}% of outflows (-25 pts)")
        elif luxury_ratio <= 0.05 and luxury_spend > 0:
            score += 10
            positive_factors.append(f"Controlled, modest luxury spending: {round(luxury_ratio * 100, 1)}% of outflows (+10 pts)")

        # (f) Savings & Investments (Saccos, MMFs, Chumz)
        if savings_ratio > 0.15:
            score += 70
            positive_factors.append(f"Strong wealth accumulation: {round(savings_ratio * 100, 1)}% of outflows directed to Saccos/MMFs (+70 pts)")
        elif savings_ratio > 0.05:
            score += 40
            positive_factors.append(f"Consistent savings habit: {round(savings_ratio * 100, 1)}% directed to investments (+40 pts)")

        # (g) Essential Expenditure Consistency (Utilities, Groceries, Rent, Education)
        if essential_ratio > 0.40:
            score += 30
            positive_factors.append(f"Stable primary commitments: {round(essential_ratio * 100, 1)}% allocated to essential needs (+30 pts)")

        if "Utilities" in category_totals:
            score += 15
            positive_factors.append("Demonstrated utility payment history (e.g., KPLC, Water, Internet) (+15 pts)")

        # (h) Debt Servicing
        if debt_service_ratio > 0.45:
            score -= 60
            risk_factors.append(f"High debt servicing burden: {round(debt_service_ratio * 100, 1)}% of outflows goes to loan repayments (-60 pts)")
            rule_flags.append("DEBT_OVERBURDENED")
        elif 0.05 <= debt_service_ratio <= 0.25:
            score += 15
            positive_factors.append(f"Active, disciplined debt servicing ({round(debt_service_ratio * 100, 1)}% of outflows) (+15 pts)")

        # 3. Final Score Clamping & Tier
        final_score = max(300, min(850, score))

        if final_score >= 750:
            tier = CreditRiskTier.EXCELLENT
            limit_multiplier = 0.45
        elif final_score >= 650:
            tier = CreditRiskTier.GOOD
            limit_multiplier = 0.30
        elif final_score >= 550:
            tier = CreditRiskTier.FAIR
            limit_multiplier = 0.15
        elif final_score >= 450:
            tier = CreditRiskTier.POOR
            limit_multiplier = 0.05
        else:
            tier = CreditRiskTier.VERY_POOR
            limit_multiplier = 0.0

        # Recommended Credit Limit based on income / monthly flow
        # In mobile money, baseline monthly velocity = max(total_inflow, total_outflow)
        base_capacity = max(total_inflow, total_outflow)
        recommended_limit = round(base_capacity * limit_multiplier, -2) if limit_multiplier > 0 else 0.0

        confidence = round(min(1.0, len(enriched_transactions) / 10.0), 2)

        return CreditScoreReport(
            credit_score=final_score,
            risk_tier=tier,
            recommended_limit_kes=recommended_limit,
            confidence=confidence,
            spending_habits=SpendingHabits(
                total_spend=round(total_outflow, 2),
                category_totals=category_totals,
                category_percentages=cat_pcts,
                essential_ratio=essential_ratio,
                luxury_ratio=luxury_ratio,
                gambling_ratio=gambling_ratio,
                debt_service_ratio=debt_service_ratio,
                savings_ratio=savings_ratio,
            ),
            cash_flow=CashFlowMetrics(
                total_inflow=round(total_inflow, 2),
                total_outflow=round(total_outflow, 2),
                net_cashflow=net_cashflow,
                inflow_outflow_ratio=inflow_outflow_ratio,
                average_tx_amount=avg_tx,
                fuliza_count=fuliza_count,
                fuliza_total_amount=round(fuliza_total, 2),
                negative_balance_count=negative_balance_count,
                latest_balance=latest_balance,
            ),
            positive_factors=positive_factors,
            risk_factors=risk_factors,
            rule_flags=rule_flags,
        )

