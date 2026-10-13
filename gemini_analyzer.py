import os
import re
import json
import base64
import requests
from typing import Dict, List, Optional, Tuple, Any

GEMINI_MODEL = "gemini-3-flash-preview"


class GeminiFinancialAnalyzer:
    """
    AI Financial Intelligence Engine powered by Google Gemini API.
    Provides deep credit underwriting summaries, actionable credit-building roadmaps
    grounded in Kenyan banking (CBK) and SACCO regulations, and multimodal OCR
    for complex or scanned bank/M-Pesa statements.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={self.api_key}"

    def analyze_statement(
        self,
        report_data: Dict[str, Any],
        transactions: List[Any],
        statement_type: str = "Bank Statement",
    ) -> Dict[str, Any]:
        """
        Generates an executive summary and strategic credit improvement action plan
        using Gemini 3.1 Flash Lite, tailored to Kenyan financial credit criteria.
        """
        cash_flow = report_data.get("cash_flow", {})
        spending = report_data.get("spending_habits", {})
        score = report_data.get("credit_score", 600)
        risk_tier = report_data.get("risk_tier", "FAIR")
        limit_kes = report_data.get("recommended_limit_kes", 0.0)

        # Build prompt context
        inflow = cash_flow.get("total_inflow", 0.0)
        outflow = cash_flow.get("total_outflow", 0.0)
        net_cf = cash_flow.get("net_cashflow", 0.0)
        fuliza_count = cash_flow.get("fuliza_count", 0)
        fuliza_amt = cash_flow.get("fuliza_total_amount", 0.0)
        balance = cash_flow.get("latest_balance", 0.0)

        cat_totals = spending.get("category_totals", {})
        top_cats = sorted(cat_totals.items(), key=lambda x: x[1], reverse=True)[:5]
        top_cats_str = ", ".join([f"{c}: KES {a:,.0f}" for c, a in top_cats]) or "None"

        essential_pct = spending.get("essential_ratio", 0.0) * 100
        savings_pct = spending.get("savings_ratio", 0.0) * 100
        luxury_pct = spending.get("luxury_ratio", 0.0) * 100
        debt_pct = spending.get("debt_service_ratio", 0.0) * 100
        gambling_pct = spending.get("gambling_ratio", 0.0) * 100

        prompt = f"""
You are the Chief Credit Underwriter at Afriqan Credit Scorer, an expert in Kenyan personal finance, Central Bank of Kenya (CBK) Prudential Guidelines, SASRA SACCO policies, and Kenyan mobile money underwriting (Fuliza, Hustler Fund, M-Shwari, Tala, Commercial Banks).

Evaluate this borrower's financial statement profile:
- Statement Type: {statement_type}
- Total Records Analyzed: {len(transactions)}
- Afriqan Credit Score: {score}/850 (Tier: {risk_tier})
- Recommended Micro-Loan Limit: KES {limit_kes:,.2f}
- Money Received (Total Inflow): KES {inflow:,.2f}
- Money Sent (Total Outflow): KES {outflow:,.2f}
- Net Cash Flow: KES {net_cf:,.2f}
- Terminal Balance: KES {balance if balance is not None else 0.0:,.2f}
- Fuliza / Overdraft Events: {fuliza_count} instances (Total KES {fuliza_amt:,.2f})
- Top Outflow Categories: {top_cats_str}
- Essential Spend Ratio: {essential_pct:.1f}%
- Savings & Investments (Sacco/MMF): {savings_pct:.1f}%
- Discretionary Luxury/Nightlife: {luxury_pct:.1f}%
- Debt Service Burden: {debt_pct:.1f}%
- Gambling / Sports Betting: {gambling_pct:.1f}%

Provide your assessment strictly in the following JSON format:
{{
  "executive_summary": "Two comprehensive, professional paragraphs. Paragraph 1 discusses the cash flow dynamics, liquidity buffers, and income regularity. Paragraph 2 evaluates behavioral risk factors including debt burden, Fuliza reliance, gambling, and overall creditworthiness under Kenyan bank and SACCO lending standards.",
  "action_plan": [
    "Actionable recommendation 1 addressing immediate credit risk or debt management",
    "Actionable recommendation 2 addressing savings or SACCO wealth accumulation (referencing SACCO 3x multiplier potential)",
    "Actionable recommendation 3 addressing cash flow optimization or mobile overdraft discipline",
    "Actionable recommendation 4 addressing utility/repayment records to unlock higher limits"
  ]
}}
Do NOT output markdown code fences (like ```json), output ONLY the raw JSON string.
"""

        try:
            resp = requests.post(
                self.base_url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.2, "maxOutputTokens": 3000},
                },
                timeout=18,
            )
            if resp.status_code == 200:
                data = resp.json()
                text_content = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
                )
                # Strip markdown blocks if present
                if "```json" in text_content:
                    text_content = text_content.split("```json")[1].split("```")[0].strip()
                elif "```" in text_content:
                    text_content = text_content.split("```")[1].split("```")[0].strip()
                
                try:
                    parsed = json.loads(text_content)
                    if "executive_summary" in parsed and "action_plan" in parsed:
                        return parsed
                except Exception:
                    # Fallback regex extraction if slight JSON syntax glitch
                    summary_match = re.search(r'"executive_summary"\s*:\s*"(.*?)(?<!\\)"', text_content, re.DOTALL)
                    plan_match = re.findall(r'"([^"\n]{10,250})"', text_content)
                    if summary_match:
                        summary_txt = summary_match.group(1).replace('\\"', '"').replace('\\n', '\n')
                        plan_items = [p for p in plan_match if p != summary_match.group(1) and not p.startswith("executive_summary") and not p.startswith("action_plan")]
                        if plan_items:
                            return {
                                "executive_summary": summary_txt,
                                "action_plan": plan_items[:5]
                            }
        except Exception as e:
            print(f"[GeminiAnalyzer] API call fallback: {e}")

        # Deterministic Kenyan Financial Expert Fallback
        return self._generate_rule_based_analysis(
            score, risk_tier, limit_kes, inflow, outflow, net_cf,
            fuliza_count, fuliza_amt, savings_pct, debt_pct, gambling_pct, top_cats_str
        )

    def _generate_rule_based_analysis(
        self, score, risk_tier, limit_kes, inflow, outflow, net_cf,
        fuliza_count, fuliza_amt, savings_pct, debt_pct, gambling_pct, top_cats_str
    ) -> Dict[str, Any]:
        """Provides expert underwriting summary when offline."""
        cf_health = "positive liquidity reserve" if net_cf >= 0 else "cash flow deficit"
        p1 = (
            f"The analyzed statement reflects a total incoming volume of KES {inflow:,.2f} versus total outflows of KES {outflow:,.2f}, "
            f"resulting in a net cash flow of KES {net_cf:,.2f} ({cf_health}). Under Central Bank of Kenya (CBK) prudential benchmarks, "
            f"the account demonstrates active financial velocity across primary expenditure categories ({top_cats_str}). "
            f"With a calculated credit score of {score}/850 placed in the {risk_tier} tier, the applicant qualifies for an initial recommended credit capacity of KES {limit_kes:,.2f}."
        )

        risk_concerns = []
        if fuliza_count > 0:
            risk_concerns.append(f"{fuliza_count} Fuliza overdraft reliance cycles (KES {fuliza_amt:,.2f})")
        if gambling_pct > 5:
            risk_concerns.append(f"heightened sports betting/gambling volume ({gambling_pct:.1f}% of outflows)")
        if debt_pct > 35:
            risk_concerns.append(f"debt service ratio above the CBK recommended 33.3% ceiling ({debt_pct:.1f}%)")

        risk_str = ", ".join(risk_concerns) if risk_concerns else "no critical behavioral red flags"

        p2 = (
            f"From an underwriting standpoint, behavioral indicators flag {risk_str}. Demonstrated savings and SACCO allocations "
            f"stand at {savings_pct:.1f}% of aggregate disbursements. Maintaining consistent non-withdrawable savings in regulated SACCOs or Money Market Funds (MMFs) "
            f"serves as a primary multiplier to scale borrowing limits up to 3x-4x savings, while minimizing short-term digital loan rollovers."
        )

        action_plan = [
            "Maintain positive net cash flow by keeping monthly outflows below 80% of aggregate monthly deposits.",
            "Eliminate or restrict Fuliza daily overdraft usage to avoid recurring daily administrative charges that depress mobile credit scores.",
            "Direct at least 15% to 20% of monthly inflows into regulated SACCO deposits (e.g. Stima, Harambee, Safaricom) or MMFs to unlock 3x-4x asset financing borrowing power.",
            "Consolidate multiple high-interest digital loan repayments into a single structured facility to reduce debt service burden below 30%.",
            "Consistently settle utility commitments (KPLC prepaid tokens, water, internet) via registered Paybill channels to establish verifiable credit payment timestamps."
        ]

        return {
            "executive_summary": f"{p1}\n\n{p2}",
            "action_plan": action_plan
        }

    def ocr_extract_table_from_text(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Extracts structured transaction rows from non-standard raw text using Gemini.
        Returns list of dicts: [{'date': 'YYYY-MM-DD', 'narrative': '...', 'amount': float, 'is_credit': bool, 'balance': float}]
        """
        if not raw_text.strip() or len(raw_text.strip()) < 30:
            return []

        prompt = f"""
You are an expert OCR financial statement parser.
The following text was extracted from a Kenyan bank or mobile money statement, but regular table regex failed to parse all rows.
Extract every financial transaction found in the text into a clean JSON array of objects.

Each transaction object must have:
- "date": Date string formatted as "YYYY-MM-DD"
- "narrative": The description, payee, or details of the transaction
- "amount": Positive float number representing the transaction amount
- "is_credit": Boolean (true if money was deposited/received/credit, false if paid out/withdrawn/debit)
- "balance": Float representing account balance after transaction, or null if not available

Raw Statement Text:
{raw_text[:6000]}

Respond ONLY with a valid JSON array of transaction objects. Do NOT include markdown code fences (```json).
"""
        try:
            resp = requests.post(
                self.base_url,
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
                },
                timeout=15,
            )
            if resp.status_code == 200:
                data = resp.json()
                text_content = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
                )
                if text_content.startswith("```"):
                    text_content = text_content.strip("`")
                    if text_content.startswith("json"):
                        text_content = text_content[4:].strip()
                rows = json.loads(text_content)
                if isinstance(rows, list):
                    return rows
        except Exception as e:
            print(f"[GeminiAnalyzer] OCR Text extraction error: {e}")
        return []

    def ocr_extract_from_page_images(self, images_bytes: List[bytes]) -> List[Dict[str, Any]]:
        """
        Extracts transaction rows directly from page image bytes using Gemini 3.1 Flash Lite multimodal vision.
        """
        if not images_bytes:
            return []

        parts: List[Dict[str, Any]] = []
        for img in images_bytes[:3]:  # process up to first 3 pages
            b64_str = base64.b64encode(img).decode("utf-8")
            parts.append({
                "inlineData": {
                    "mimeType": "image/jpeg",
                    "data": b64_str
                }
            })

        parts.append({
            "text": """
You are an OCR Financial Parser for Kenyan bank statements (Equity, KCB, Co-op, NCBA, Stanbic, Absa, DTB) and Safaricom M-Pesa statements.
Extract all transaction entries from these document images into a structured JSON array.
Each item must have:
- "date": "YYYY-MM-DD"
- "narrative": Details / description / counterparty
- "amount": positive float
- "is_credit": boolean (true for deposits/inflows, false for withdrawals/outflows)
- "balance": float or null

Output ONLY the raw JSON array.
"""
        })

        try:
            resp = requests.post(
                self.base_url,
                json={
                    "contents": [{"parts": parts}],
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 2048},
                },
                timeout=20,
            )
            if resp.status_code == 200:
                data = resp.json()
                text_content = (
                    data.get("candidates", [{}])[0]
                    .get("content", {})
                    .get("parts", [{}])[0]
                    .get("text", "")
                    .strip()
                )
                if text_content.startswith("```"):
                    text_content = text_content.strip("`")
                    if text_content.startswith("json"):
                        text_content = text_content[4:].strip()
                rows = json.loads(text_content)
                if isinstance(rows, list):
                    return rows
        except Exception as e:
            print(f"[GeminiAnalyzer] Multimodal OCR error: {e}")
        return []
