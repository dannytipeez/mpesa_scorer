import argparse
import getpass
import json
import os
import sys
import joblib
from mpesa_pdf_parser import MpesaPdfParser
from credit_scorer import MpesaCreditScorer
import app as app_mod


def main():
    parser = argparse.ArgumentParser(
        description="M-Pesa Statement PDF Terminal Analyzer & Credit Scorer"
    )
    parser.add_argument("pdf_path", help="Path to your M-Pesa statement PDF file")
    parser.add_argument(
        "-p", "--password",
        help="Statement password (usually your National ID / Passport number). If omitted and required, you will be prompted securely.",
        default=None
    )
    parser.add_argument(
        "--show-txs", action="store_true",
        help="Print the full extracted transaction table in terminal"
    )
    parser.add_argument(
        "-o", "--output",
        help="Optional path to save results as a JSON file",
        default=None
    )

    args = parser.parse_args()

    if not os.path.exists(args.pdf_path):
        print(f"Error: File not found: '{args.pdf_path}'")
        sys.exit(1)

    # 1. Load ML Model
    model_path = "mpesa_classifier.joblib"
    if os.path.exists(model_path):
        app_mod.ml_model = joblib.load(model_path)
    else:
        print("Note: 'mpesa_classifier.joblib' not found, running on deterministic rules.")

    pdf_parser = MpesaPdfParser()
    password = args.password

    # 2. Parse PDF (with retry prompt if password missing/incorrect)
    try:
        transactions = pdf_parser.parse_pdf_file(args.pdf_path, password=password)
    except ValueError as ve:
        err_msg = str(ve).lower()
        if "password" in err_msg:
            print("This PDF is password-protected by Safaricom.")
            password = getpass.getpass("Enter M-Pesa PDF Password (National ID / Passport No): ")
            try:
                transactions = pdf_parser.parse_pdf_file(args.pdf_path, password=password)
            except Exception as e:
                print(f"Error: {e}")
                sys.exit(1)
        else:
            print(f"Error: {ve}")
            sys.exit(1)
    except Exception as e:
        print(f"Error opening PDF: {e}")
        sys.exit(1)

    if not transactions:
        print("Warning: No transactions found in the provided PDF statement.")
        sys.exit(0)

    # 3. Classify Transactions
    enriched_txs = []
    for tx in transactions:
        cat, conf, engine = app_mod.classify_entity(tx.counterparty, tx.account_reference)
        enriched_txs.append({
            "parsed": tx,
            "category": cat,
            "confidence": conf,
            "decision_engine": engine,
        })

    # 4. Run Credit Scorer
    scorer = MpesaCreditScorer()
    report = scorer.score_transactions(enriched_txs)

    # 5. Print Terminal Output
    print("\n" + "=" * 65)
    print("      M-PESA STATEMENT ANALYSIS & CREDIT RISK REPORT")
    print("=" * 65)
    print(f"Statement File : {os.path.basename(args.pdf_path)}")
    print(f"Transactions   : {len(transactions)} extracted")
    print("-" * 65)

    # Credit Score Summary
    print(f"CREDIT SCORE   : {report.credit_score} / 850 ({report.risk_tier.value})")
    print(f"RECOMMENDED LOAN LIMIT : KES {report.recommended_limit_kes:,.2f}")
    print(f"SCORING CONFIDENCE     : {report.confidence * 100:.0f}%")
    print("-" * 65)

    # Cash Flow Summary
    cf = report.cash_flow
    print("CASH FLOW & LIQUIDITY METRICS:")
    print(f"  • Total Inflows       : KES {cf.total_inflow:12,.2f}")
    print(f"  • Total Outflows      : KES {cf.total_outflow:12,.2f}")
    print(f"  • Net Cash Flow       : KES {cf.net_cashflow:12,.2f} {'(Positive)' if cf.net_cashflow >= 0 else '(Deficit)'}")
    print(f"  • Fuliza Overdrafts   : {cf.fuliza_count} instances (KES {cf.fuliza_total_amount:,.2f})")
    bal_str = f"KES {cf.latest_balance:,.2f}" if cf.latest_balance is not None else "N/A"
    print(f"  • Terminal Balance    : {bal_str}")
    print("-" * 65)

    # Spending Habits Breakdown
    sh = report.spending_habits
    print("SPENDING HABITS BREAKDOWN:")
    for cat, pct in sorted(sh.category_percentages.items(), key=lambda x: x[1], reverse=True):
        amt = sh.category_totals.get(cat, 0.0)
        print(f"  • {cat:24s}: KES {amt:10,.2f} ({pct:5.1f}%)")

    print("\nBEHAVIORAL RATIOS:")
    print(f"  • Essential Needs (Utilities, Groceries, Rent, Edu) : {sh.essential_ratio * 100:.1f}%")
    print(f"  • Savings & Investments (MMF, Saccos)              : {sh.savings_ratio * 100:.1f}%")
    print(f"  • Luxury & Nightlife (Wines, Bars, Lounges)        : {sh.luxury_ratio * 100:.1f}%")
    print(f"  • Debt Servicing Burden (Loan Repayments)          : {sh.debt_service_ratio * 100:.1f}%")
    print(f"  • Gambling / Sports Betting Spend                  : {sh.gambling_ratio * 100:.1f}%")
    print("-" * 65)

    # Positive Factors
    print("POSITIVE SCORING FACTORS:")
    if report.positive_factors:
        for p in report.positive_factors:
            print(f"  [+] {p}")
    else:
        print("  None detected.")

    # Risk Factors & Flags
    print("\nRISK FACTORS & OVERDRAFT FLAGS:")
    if report.risk_factors:
        for r in report.risk_factors:
            print(f"  [-] {r}")
    else:
        print("  Zero risk factors detected.")

    if report.rule_flags:
        print(f"  Flags: {', '.join(report.rule_flags)}")
    print("=" * 65)

    # Optional Table
    if args.show_txs:
        print("\nCLASSIFIED TRANSACTIONS TABLE:")
        print(f"{'Receipt':12s} | {'Type':13s} | {'Counterparty':30s} | {'Category':20s} | {'Amount (KES)':12s}")
        print("-" * 95)
        for t in enriched_txs:
            p = t["parsed"]
            cp = p.counterparty_masked[:28]
            print(f"{p.transaction_id:12s} | {p.tx_type.value:13s} | {cp:30s} | {t['category']:20s} | {p.amount:12,.2f}")
        print("=" * 95)

    # Optional Export JSON
    if args.output:
        out_data = {
            "report": report.model_dump(),
            "transactions": [
                {
                    "receipt": t["parsed"].transaction_id,
                    "type": t["parsed"].tx_type.value,
                    "counterparty": t["parsed"].counterparty,
                    "counterparty_masked": t["parsed"].counterparty_masked,
                    "amount": t["parsed"].amount,
                    "balance": t["parsed"].balance,
                    "category": t["category"],
                    "confidence": t["confidence"],
                }
                for t in enriched_txs
            ]
        }
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, default=str)
        print(f"\nSaved analysis JSON report to: {args.output}")


if __name__ == "__main__":
    main()

