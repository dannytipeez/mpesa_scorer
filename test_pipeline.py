import json
import joblib
from mpesa_parser import MpesaParser
from credit_scorer import MpesaCreditScorer
import app

def test_full_pipeline():
    # Load model manually for test run
    app.ml_model = joblib.load("mpesa_classifier.joblib")
    parser = MpesaParser()
    scorer = MpesaCreditScorer()

    sample_messages = [
        # 1. Paybill (Utilities)
        "SH2819DK3L Confirmed. Ksh1,500.00 paid to KPLC PREPAID. for account 482910394 on 12/03/26 at 4:15 PM. New M-PESA balance is Ksh4,210.00. Transaction cost, Ksh23.00.",
        
        # 2. Buy Goods with Fuliza (Groceries)
        "RJ99201LMA Confirmed. Ksh2,450.00 paid to QUICKMART TOM MBOYA NAIROBI. on 14/05/2026 at 8:30 PM. New M-PESA balance is Ksh-300.00. Fuliza M-PESA amount is Ksh300.00. Transaction cost, Ksh0.00.",
        
        # 3. P2P Send with phone masking (Personal Transfers)
        "TK1029384C Confirmed. Ksh500.00 sent to JOHN KAMAU 0722112233 on 10/01/26 at 11:00 AM. New M-PESA balance is Ksh1,200.00. Transaction cost, Ksh7.00.",
        
        # 4. Airtime (Telecommunications)
        "PL4820194A Confirmed. You bought Ksh100.00 of airtime on 02/02/26 at 9:15 AM. New M-PESA balance is Ksh1,100.00. Transaction cost, Ksh0.00.",
        
        # 5. Wines (Luxury)
        "LQ8912301A Confirmed. Ksh3,200.00 paid to OAKS AND CORKS WINES AND SPIRITS. on 15/05/26 at 9:45 PM. New M-PESA balance is Ksh8,500.00. Transaction cost, Ksh0.00.",
        
        # 6. Lounge (Luxury)
        "NV7718290B Confirmed. Ksh4,500.00 paid to QUIVER LOUNGE THIKA ROAD. on 16/05/26 at 11:30 PM. New M-PESA balance is Ksh4,000.00. Transaction cost, Ksh0.00.",
        
        # 7. Bar (Luxury)
        "BR6628190X Confirmed. Ksh2,800.00 paid to THE ALCHEMIST BAR WESTLANDS. on 17/05/26 at 10:15 PM. New M-PESA balance is Ksh1,200.00. Transaction cost, Ksh0.00.",

        # 8. Food: Restaurant (Food & Dining)
        "FD1029384A Confirmed. Ksh1,850.00 paid to JAVA HOUSE VALLEY ARCADE. on 18/05/26 at 1:15 PM. New M-PESA balance is Ksh9,350.00. Transaction cost, Ksh0.00.",

        # 9. Food: Fast Food & Grill (Food & Dining)
        "FD9928172B Confirmed. Ksh1,200.00 paid to KFC KIMATHI STREET NAIROBI. on 19/05/26 at 7:45 PM. New M-PESA balance is Ksh8,150.00. Transaction cost, Ksh0.00.",

        # 10. Food: Nyama Choma / Charcoal Grill (Food & Dining)
        "FD5544332C Confirmed. Ksh2,100.00 paid to CITY NYAMA CHOMA AND CHARCOAL GRILL. on 20/05/26 at 8:00 PM. New M-PESA balance is Ksh6,050.00. Transaction cost, Ksh0.00.",

        # 11. Savings & Investment (Savings & Investments)
        "SV1928374C Confirmed. Ksh10,000.00 paid to CIC MONEY MARKET FUND MMF. for account 819201 on 21/05/26 at 10:00 AM. New M-PESA balance is Ksh14,000.00. Transaction cost, Ksh34.00.",
        
        # 12. Loan Repayment (Loan Repayment)
        "LN9182736D Confirmed. Ksh2,000.00 paid to TALA KENYA MOBILE LOAN. for account 0722112233 on 22/05/26 at 2:00 PM. New M-PESA balance is Ksh12,000.00. Transaction cost, Ksh15.00.",
        
        # 13. Inflow: Received Money (Salary / Revenue)
        "RC0192837E Confirmed. You have received Ksh60,000.00 from ACME TECH SOLUTIONS LTD on 25/05/26 at 8:00 AM. New M-PESA balance is Ksh72,000.00. Transaction cost, Ksh0.00.",
    ]

    enriched_txs = []
    print("=== TRANSACTION EXTRACTION & CATEGORIZATION ===")
    for raw in sample_messages:
        parsed = parser.parse(raw)
        assert parsed is not None, f"Failed on: {raw}"
        
        cat, conf, engine = app.classify_entity(parsed.counterparty, parsed.account_reference)
        enriched_txs.append({
            "parsed": parsed,
            "category": cat,
            "confidence": conf,
            "decision_engine": engine,
        })
        
        print(f"\n[ID: {parsed.transaction_id}] Type: {parsed.tx_type}")
        print(f"Counterparty: {parsed.counterparty} -> Masked: {parsed.counterparty_masked}")
        print(f"Amount: KES {parsed.amount:,.2f} | Balance: KES {parsed.balance} | Fuliza: {parsed.fuliza_amount}")
        print(f"Categorization: {cat} (Confidence: {conf * 100:.1f}%, Engine: {engine})")

    # Assertions on Luxury mapping (wines, bar, lounge)
    luxury_txs = [t for t in enriched_txs if t["category"] == "Luxury"]
    assert len(luxury_txs) == 3, f"Expected 3 Luxury transactions, got {len(luxury_txs)}"
    for t in luxury_txs:
        assert any(term in t["parsed"].counterparty.lower() for term in ["wine", "bar", "lounge"]), \
            f"Non-wine/bar/lounge entity in Luxury: {t['parsed'].counterparty}"

    # Assertions on Food & Dining
    food_txs = [t for t in enriched_txs if t["category"] == "Food & Dining"]
    assert len(food_txs) == 3, f"Expected 3 Food & Dining transactions, got {len(food_txs)}"

    # Run Credit Scorer
    print("\n" + "=" * 50)
    print("=== M-PESA CREDIT SCORING & BEHAVIORAL PROFILE ===")
    print("=" * 50)
    report = scorer.score_transactions(enriched_txs)

    print(f"Credit Score: {report.credit_score} / 850 ({report.risk_tier.value})")
    print(f"Recommended Credit Limit: KES {report.recommended_limit_kes:,.2f}")
    print(f"Confidence: {report.confidence * 100:.0f}%")
    
    print("\n--- Spending Habits Breakdown ---")
    print(f"Total Outflows: KES {report.spending_habits.total_spend:,.2f}")
    for cat, pct in report.spending_habits.category_percentages.items():
        amt = report.spending_habits.category_totals.get(cat, 0.0)
        print(f"  - {cat:22s}: KES {amt:9,.2f} ({pct:5.1f}%)")

    print("\n--- Key Financial Ratios ---")
    print(f"  - Essential Spend Ratio : {report.spending_habits.essential_ratio * 100:.1f}%")
    print(f"  - Food & Dining         : {(report.spending_habits.category_totals.get('Food & Dining', 0) / report.spending_habits.total_spend) * 100:.1f}%")
    print(f"  - Luxury (Wines/Bar/Lounge): {report.spending_habits.luxury_ratio * 100:.1f}%")
    print(f"  - Savings & Investments : {report.spending_habits.savings_ratio * 100:.1f}%")
    print(f"  - Debt Servicing Ratio  : {report.spending_habits.debt_service_ratio * 100:.1f}%")
    print(f"  - Gambling Ratio        : {report.spending_habits.gambling_ratio * 100:.1f}%")

    print("\n--- Positive Factors ---")
    for f in report.positive_factors:
        print(f"  [+] {f}")

    print("\n--- Risk Factors & Flags ---")
    for f in report.risk_factors:
        print(f"  [-] {f}")
    if report.rule_flags:
        print(f"  Flags: {', '.join(report.rule_flags)}")

    print("\nPipeline and Credit Scorer verification passed successfully!")


if __name__ == "__main__":
    test_full_pipeline()
