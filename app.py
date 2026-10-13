import io
import os
import re
from contextlib import asynccontextmanager
from typing import List, Optional
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from pydantic import BaseModel
from mpesa_parser import MpesaParser, ParsedTransaction
from mpesa_pdf_parser import MpesaPdfParser
from bank_statement_parser import BankStatementParser
from credit_scorer import MpesaCreditScorer, CreditScoreReport
from mpesa_gateway import MpesaGateway

# In-memory model handle
ml_model = None
mpesa_gateway = MpesaGateway()

# Regex rule: Map anything containing "wines", "wine", "bar", or "lounge" as discrete words to Luxury
LUXURY_PATTERN = re.compile(r"\b(wines?|bar|lounge)\b", re.IGNORECASE)

# Deterministic fast-path lookup table for food, grocery, till, and paybill names
DETERMINISTIC_RULES = {
    # Food & Dining (Restaurants, Cafes, Eateries, Fast Food, Bakeries, Grills)
    "java house": "Food & Dining",
    "artcaffe": "Food & Dining",
    "kfc": "Food & Dining",
    "dominos": "Food & Dining",
    "cj restaurant": "Food & Dining",
    "cj's": "Food & Dining",
    "big square": "Food & Dining",
    "galitos": "Food & Dining",
    "pizza inn": "Food & Dining",
    "chicken inn": "Food & Dining",
    "creamy inn": "Food & Dining",
    "subway": "Food & Dining",
    "steers": "Food & Dining",
    "debonairs": "Food & Dining",
    "restaurant": "Food & Dining",
    "cafe": "Food & Dining",
    "café": "Food & Dining",
    "bakery": "Food & Dining",
    "bistro": "Food & Dining",
    "grill": "Food & Dining",
    "diner": "Food & Dining",
    "eatery": "Food & Dining",
    "canteen": "Food & Dining",
    "kitchen": "Food & Dining",
    "fast food": "Food & Dining",
    "nyama choma": "Food & Dining",
    "choma": "Food & Dining",
    "roast": "Food & Dining",
    "pizza": "Food & Dining",
    "burger": "Food & Dining",
    "coffee": "Food & Dining",
    "catering": "Food & Dining",
    "fish point": "Food & Dining",
    "swahili dishes": "Food & Dining",
    "food court": "Food & Dining",
    "delight foods": "Food & Dining",

    # Groceries (Supermarkets, Greengrocers, Markets, Butcheries)
    "naivas": "Groceries",
    "quickmart": "Groceries",
    "carrefour": "Groceries",
    "chandarana": "Groceries",
    "cleanshelf": "Groceries",
    "eastmatt": "Groceries",
    "foodplus": "Groceries",
    "supermarket": "Groceries",
    "grocery": "Groceries",
    "groceries": "Groceries",
    "mama mboga": "Groceries",
    "vegetables": "Groceries",
    "veggies": "Groceries",
    "greengrocer": "Groceries",
    "fruits": "Groceries",
    "butchery": "Groceries",
    "fresh meat": "Groceries",
    "mini market": "Groceries",
    "wholesalers": "Groceries",

    # Utilities (Essential Fixed Overhead)
    "kplc": "Utilities",
    "kenya power": "Utilities",
    "nairobi water": "Utilities",
    "water and sewerage": "Utilities",
    "sewerage": "Utilities",
    "zuku": "Utilities",
    "safaricom home": "Utilities",
    "faiba": "Utilities",
    "prepaid meter": "Utilities",

    # Transport (Fuel Stations, Commute, Matatus)
    "uber": "Transport",
    "bolt": "Transport",
    "little cab": "Transport",
    "totalenergies": "Transport",
    "total energy": "Transport",
    "rubis": "Transport",
    "shell": "Transport",
    "ola energy": "Transport",
    "astrol": "Transport",
    "petrol": "Transport",
    "fuel": "Transport",
    "station": "Transport",
    "matatu": "Transport",
    "super metro": "Transport",
    "2nk sacco": "Transport",
    "shuttle": "Transport",
    "bus fare": "Transport",

    # Telecommunications
    "safaricom airtime": "Telecommunications",
    "airtel money airtime": "Telecommunications",
    "airtime topup": "Telecommunications",
    "postpaid bill": "Telecommunications",

    # Healthcare (Emergency & Health)
    "aga khan": "Healthcare",
    "gertrudes": "Healthcare",
    "nairobi hospital": "Healthcare",
    "mater hospital": "Healthcare",
    "mp shah": "Healthcare",
    "pharmacy": "Healthcare",
    "chemist": "Healthcare",
    "clinic": "Healthcare",
    "dispensary": "Healthcare",
    "laboratory": "Healthcare",
    "medplus": "Healthcare",
    "haltons": "Healthcare",
    "healthcare": "Healthcare",
    "medical": "Healthcare",

    # Gambling & Betting (High Credit Risk Indicator)
    "sportpesa": "Gambling",
    "betika": "Gambling",
    "odibets": "Gambling",
    "mozzart": "Gambling",
    "1xbet": "Gambling",
    "shabiki": "Gambling",
    "betway": "Gambling",
    "betting": "Gambling",
    "casino": "Gambling",
    "sportsbook": "Gambling",
    "lotto": "Gambling",

    # Loan Repayments (Debt Servicing)
    "hustler fund": "Loan Repayment",
    "tala": "Loan Repayment",
    "branch international": "Loan Repayment",
    "branch loan": "Loan Repayment",
    "zenka": "Loan Repayment",
    "timiza": "Loan Repayment",
    "m-shwari loan": "Loan Repayment",
    "kcb mpesa loan": "Loan Repayment",
    "okash": "Loan Repayment",
    "platinum credit": "Loan Repayment",
    "ipesa": "Loan Repayment",

    # Savings & Investments (Positive Wealth Accumulation)
    "money market": "Savings & Investments",
    "mmf": "Savings & Investments",
    "chumz": "Savings & Investments",
    "sanlam": "Savings & Investments",
    "britam": "Savings & Investments",
    "cic asset": "Savings & Investments",
    "cic money market": "Savings & Investments",
    "stima sacco": "Savings & Investments",
    "harambee sacco": "Savings & Investments",
    "sacco deposit": "Savings & Investments",
    "sacco shares": "Savings & Investments",
    "mali wealth": "Savings & Investments",
    "unit trust": "Savings & Investments",

    # Rent & Housing (Fixed Living Expenses)
    "rent": "Rent & Housing",
    "apartments": "Rent & Housing",
    "estate management": "Rent & Housing",
    "housing": "Rent & Housing",
    "hass consult": "Rent & Housing",
    "lloyd masika": "Rent & Housing",

    # Education (Tuition & School Fees)
    "school fees": "Education",
    "tuition": "Education",
    "academy": "Education",
    "strathmore": "Education",
    "makini school": "Education",
    "brookhouse": "Education",
    "university of nairobi": "Education",
    "kenyatta university": "Education",

    # Bank Charges & Administrative Fees
    "bank charges": "General",
    "maintenance fee": "General",
    "ledger fee": "General",
    "excise duty": "General",
    "withholding tax": "General",
}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global ml_model
    model_path = "mpesa_classifier.joblib"
    if os.path.exists(model_path):
        ml_model = joblib.load(model_path)
        print("Loaded ML classifier pipeline successfully.")
    else:
        print("WARNING: 'mpesa_classifier.joblib' not found. Train via train_classifier.py.")
    yield


app = FastAPI(
    title="M-Pesa Extraction, Categorization & Credit Scoring Engine",
    version="1.3.0",
    lifespan=lifespan,
)

parser = MpesaParser()
credit_scorer = MpesaCreditScorer()


class SingleSMSRequest(BaseModel):
    sms_text: str


class BatchSMSRequest(BaseModel):
    messages: List[str]


class EnrichedTransaction(BaseModel):
    parsed: ParsedTransaction
    category: str
    confidence: float
    decision_engine: str  # "RULE_BASED" or "ML_MODEL"


class FullScoringResponse(BaseModel):
    report: CreditScoreReport
    transactions: List[EnrichedTransaction]


def classify_entity(counterparty: str, acc_ref: Optional[str]) -> tuple[str, float, str]:
    text_repr = f"{counterparty} {acc_ref or ''}".lower().strip()

    # 1. Strict Luxury mapping: names with wines, wine, bar, or lounge
    if LUXURY_PATTERN.search(text_repr):
        return "Luxury", 1.0, "RULE_BASED"

    # 2. Deterministic fast-path for food, grocery, till, and paybill names
    for keyword, cat in DETERMINISTIC_RULES.items():
        if keyword in text_repr:
            return cat, 1.0, "RULE_BASED"

    # 3. Machine Learning Pipeline Fallback
    if ml_model is not None:
        try:
            pred = ml_model.predict([text_repr])[0]
            probs = ml_model.predict_proba([text_repr])[0]
            confidence = float(np.max(probs))
            return pred, round(confidence, 4), "ML_MODEL"
        except Exception:
            pass

    return "Uncategorized", 0.0, "FALLBACK"


@app.get("/", response_class=FileResponse)
def serve_dashboard():
    """Serves the interactive web UI dashboard."""
    template_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if os.path.exists(template_path):
        return FileResponse(template_path)
    raise HTTPException(status_code=404, detail="Dashboard UI template not found.")


@app.post("/api/v1/parse", response_model=EnrichedTransaction)
def parse_single_message(payload: SingleSMSRequest):
    tx = parser.parse(payload.sms_text)
    if not tx:
        raise HTTPException(
            status_code=422,
            detail="Unable to parse message. Format does not match supported M-Pesa receipts."
        )

    cat, conf, engine = classify_entity(tx.counterparty, tx.account_reference)
    return EnrichedTransaction(
        parsed=tx,
        category=cat,
        confidence=conf,
        decision_engine=engine
    )


@app.post("/api/v1/parse-batch", response_model=List[EnrichedTransaction])
def parse_batch_messages(payload: BatchSMSRequest):
    results = []
    for raw in payload.messages:
        tx = parser.parse(raw)
        if tx:
            cat, conf, engine = classify_entity(tx.counterparty, tx.account_reference)
            results.append(
                EnrichedTransaction(
                    parsed=tx,
                    category=cat,
                    confidence=conf,
                    decision_engine=engine
                )
            )
    return results


@app.post("/api/v1/score", response_model=CreditScoreReport)
def calculate_credit_score(payload: BatchSMSRequest):
    """
    Parses a batch of M-Pesa messages, enriches each with spending category,
    and calculates an end-to-end credit score with risk tiers and limits.
    """
    enriched_txs = []
    for raw in payload.messages:
        tx = parser.parse(raw)
        if tx:
            cat, conf, engine = classify_entity(tx.counterparty, tx.account_reference)
            enriched_txs.append({
                "parsed": tx,
                "category": cat,
                "confidence": conf,
                "decision_engine": engine,
            })

    report = credit_scorer.score_transactions(enriched_txs)
    return report


@app.post("/api/v1/score-full", response_model=FullScoringResponse)
def calculate_credit_score_full(payload: BatchSMSRequest):
    """
    Parses messages and returns both the CreditScoreReport and all enriched transactions
    for interactive dashboard rendering.
    """
    enriched_list: List[EnrichedTransaction] = []
    raw_dicts = []

    for raw in payload.messages:
        tx = parser.parse(raw)
        if tx:
            cat, conf, engine = classify_entity(tx.counterparty, tx.account_reference)
            en = EnrichedTransaction(
                parsed=tx,
                category=cat,
                confidence=conf,
                decision_engine=engine,
            )
            enriched_list.append(en)
            raw_dicts.append({
                "parsed": tx,
                "category": cat,
                "confidence": conf,
                "decision_engine": engine,
            })

    report = credit_scorer.score_transactions(raw_dicts)
    return FullScoringResponse(report=report, transactions=enriched_list)


@app.post("/api/v1/upload-pdf", response_model=FullScoringResponse)
async def upload_pdf_statement(
    file: UploadFile = File(...),
    password: Optional[str] = Form(None)
):
    """
    Uploads an official Safaricom M-Pesa statement PDF (with optional National ID password),
    extracts all transactions, categorizes them, and returns credit scoring metrics and table data.
    """
    try:
        content = await file.read()
        pdf_parser = MpesaPdfParser()
        parsed_txs = pdf_parser.parse_pdf_stream(io.BytesIO(content), password=password)

        if not parsed_txs:
            raise HTTPException(
                status_code=422,
                detail="No transactions could be extracted from this PDF. Ensure it is an official Safaricom e-statement."
            )

        enriched_list: List[EnrichedTransaction] = []
        raw_dicts = []

        for tx in parsed_txs:
            cat, conf, engine = classify_entity(tx.counterparty, tx.account_reference)
            en = EnrichedTransaction(
                parsed=tx,
                category=cat,
                confidence=conf,
                decision_engine=engine,
            )
            enriched_list.append(en)
            raw_dicts.append({
                "parsed": tx,
                "category": cat,
                "confidence": conf,
                "decision_engine": engine,
            })

        report = credit_scorer.score_transactions(raw_dicts)
        return FullScoringResponse(report=report, transactions=enriched_list)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF parsing error: {str(e)}")


@app.post("/api/v1/upload-bank", response_model=FullScoringResponse)
async def upload_bank_statement(
    file: UploadFile = File(...),
    password: Optional[str] = Form(None)
):
    """
    Uploads an official Kenyan Bank Statement (PDF or CSV) from Equity, KCB, Co-op,
    NCBA, Stanbic, Absa, DTB, StanChart, I&M, etc. (with optional National ID or KRA PIN password).
    """
    try:
        content = await file.read()
        bank_parser = BankStatementParser()
        filename_lower = (file.filename or "").lower()

        if filename_lower.endswith(".csv"):
            text_content = content.decode("utf-8", errors="ignore")
            parsed_txs = bank_parser.parse_csv_stream(text_content)
        else:
            parsed_txs = bank_parser.parse_pdf_stream(io.BytesIO(content), password=password)

        if not parsed_txs:
            raise HTTPException(
                status_code=422,
                detail="No transactions could be extracted from this bank statement. Please verify the format."
            )

        enriched_list: List[EnrichedTransaction] = []
        raw_dicts = []

        for tx in parsed_txs:
            cat, conf, engine = classify_entity(tx.counterparty, tx.account_reference)
            en = EnrichedTransaction(
                parsed=tx,
                category=cat,
                confidence=conf,
                decision_engine=engine,
            )
            enriched_list.append(en)
            raw_dicts.append({
                "parsed": tx,
                "category": cat,
                "confidence": conf,
                "decision_engine": engine,
            })

        report = credit_scorer.score_transactions(raw_dicts)
        return FullScoringResponse(report=report, transactions=enriched_list)

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bank statement parsing error: {str(e)}")


class StkPushRequest(BaseModel):
    phone: str
    reference: Optional[str] = "CreditScore"


@app.post("/api/v1/mpesa/stk-push")
def initiate_stk_payment(payload: StkPushRequest):
    """
    Triggers Lipa na M-Pesa Online (STK Push) on the user's phone for 50 KES.
    """
    try:
        res = mpesa_gateway.initiate_stk_push(payload.phone, reference=payload.reference)
        return res
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"M-Pesa STK Push error: {str(e)}")


@app.post("/api/v1/mpesa/callback")
def mpesa_callback(payload: dict):
    """
    Asynchronous callback webhook invoked by Safaricom Daraja API upon payment completion.
    """
    result = mpesa_gateway.process_callback(payload)
    return {"ResultCode": 0, "ResultDesc": "Callback accepted successfully."}


@app.get("/api/v1/mpesa/check-payment/{checkout_request_id}")
def check_payment_status(checkout_request_id: str):
    """
    Checks if payment has been confirmed for a given CheckoutRequestID.
    """
    status = mpesa_gateway.query_status(checkout_request_id)
    return status


@app.post("/api/v1/mpesa/demo-approve/{checkout_request_id}")
def demo_approve_payment(checkout_request_id: str):
    """
    Simulates successful payment confirmation for demonstration and testing.
    """
    return mpesa_gateway.simulate_mock_approval(checkout_request_id)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": ml_model is not None,
        "service": "mpesa-parser-api",
        "mpesa_shortcode": mpesa_gateway.shortcode,
        "mpesa_env": mpesa_gateway.env
    }

