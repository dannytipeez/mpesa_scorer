import csv
import hashlib
import io
import re
from datetime import datetime
from typing import List, Optional, Tuple
from pypdf import PdfReader
from mpesa_parser import ParsedTransaction, TransactionType
from gemini_analyzer import GeminiFinancialAnalyzer


class BankStatementParser:
    """
    Parses official Kenyan bank e-statements (PDFs & CSVs) from top banks:
    Equity Bank, KCB Bank, Co-op Bank, NCBA, Stanbic, Absa, Standard Chartered, DTB, I&M, etc.
    Supports password-protected PDFs (National ID or KRA PIN) and extracts POS, EFT, PesaLink,
    ATM, Standing Orders, and Mobile Banking transactions.
    """

    def __init__(self, salt: str = "bank_local_salt_123"):
        self.salt = salt

        # Date pattern: matches YYYY-MM-DD, DD/MM/YYYY, DD-MM-YYYY, or DD-Mon-YYYY (e.g. 15-Mar-2026)
        self.date_pattern = (
            r"(?P<date>\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}-[A-Za-z]{3}-\d{2,4})"
        )

        # Line regex for PDF tables: Date, Narrative/Description, Debit, Credit, Balance
        self.tx_line_regex = re.compile(
            rf"^{self.date_pattern}\s+(?P<rest>.+)$",
            re.IGNORECASE,
        )

        # Regex for amounts at the end of the line: [Debit] [Credit] [Balance]
        self.amounts_regex = re.compile(
            r"(?P<amount1>[\d,]+\.\d{2}|-)\s+"
            r"(?P<amount2>[\d,]+\.\d{2}|-)?\s*"
            r"(?P<balance>-?[\d,]+\.\d{2})$",
            re.IGNORECASE,
        )

    def _mask_account_or_phone(self, text: str) -> str:
        """Hashes bank accounts and phone numbers with salt for privacy."""
        def replace_phone(match):
            val = match.group(0)
            hashed = hashlib.sha256((val + self.salt).encode()).hexdigest()[:10]
            return f"[PHONE_ID_{hashed}]"

        def replace_acc(match):
            val = match.group(0)
            hashed = hashlib.sha256((val + self.salt).encode()).hexdigest()[:10]
            return f"[ACC_ID_{hashed}]"

        # Phones (07xx, 01xx, 254xx)
        text = re.sub(r"(?:\+?254|0)[17]\d{8}", replace_phone, text)
        # Bank account numbers (10 to 14 digits)
        text = re.sub(r"\b\d{10,14}\b", replace_acc, text)
        return text

    def _parse_date(self, date_str: str) -> datetime:
        d = date_str.strip()
        formats = [
            "%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d-%b-%Y", "%d-%b-%y",
            "%d/%m/%y", "%Y/%m/%d", "%d %b %Y"
        ]
        for fmt in formats:
            try:
                return datetime.strptime(d, fmt)
            except ValueError:
                pass
        return datetime.now()

    def _interpret_narrative(
        self, narrative: str
    ) -> Tuple[TransactionType, str, Optional[str]]:
        """
        Interprets bank narrative into (TransactionType, Clean Counterparty, Account Reference)
        """
        clean = " ".join(narrative.strip().split())
        c_lower = clean.lower()

        # 1. Bank Charges, SMS fees, Alerts, Ledger fees, Commissions (Utilities / Fees)
        if any(chg in c_lower for chg in [
            "sms charge", "sms fee", "alert charge", "ledger fee", "maintenance fee",
            "excise duty", "comm ", "commission", "tpg comm", "transaction charge"
        ]):
            return TransactionType.PAYBILL, "Bank Service & SMS Fee", None

        # 2. Mobile Banking / M-Pesa App Outward or Inward (e.g. APP/MPESA/254768885233/A485B18BD9DE6/ FAUSTINA NDU)
        m_app = re.search(r"app/mpesa/\d+/[A-Z0-9]+/\s*([^/]+?)(?:\s+[A-Z0-9]{8,}|$)", clean, re.IGNORECASE)
        if m_app:
            return TransactionType.SEND_MONEY, m_app.group(1).strip(), None

        m_mpesa_out = re.search(r"mpesa/(?:outward/)?(?:[0-9]+/)?(?P<entity>[^/]+)", clean, re.IGNORECASE)
        if m_mpesa_out:
            return TransactionType.PAYBILL, m_mpesa_out.group("entity").strip(), None

        # 3. PesaLink / EFT / RTGS Outward or Inward
        if any(kw in c_lower for kw in ["pesalink", "eft", "rtgs"]):
            m_pl = re.search(r"pesalink\s*(?:transfer|txn)?\s*(?:to|from)?\s*([^/]+?)(?:/\d+|$)", clean, re.IGNORECASE)
            pl_entity = m_pl.group(1).strip() if m_pl else clean
            if any(in_kw in c_lower for in_kw in ["received", "inward", "credit", "cr to", "salary"]):
                return TransactionType.RECEIVE_MONEY, f"PesaLink: {pl_entity}", None
            else:
                return TransactionType.SEND_MONEY, f"PesaLink: {pl_entity}", None

        # 4. Clean TPG and internal bank routing prefixes (e.g. TPG 49AP5E7477/KCB/Palm Limited Op/8D80E25E7477/1)
        if clean.upper().startswith("TPG"):
            cleaned_tpg = re.sub(r"^TPG\s+[A-Z0-9]+/(?:[A-Z0-9]+/)?", "", clean, flags=re.IGNORECASE)
            cleaned_tpg = re.sub(r"/[A-Z0-9]+(?:/\d+)?$", "", cleaned_tpg)
            clean = " ".join(cleaned_tpg.split())
            c_lower = clean.lower()

        # 5. POS Purchases (Cards at merchants / till)
        if "pos purchase" in c_lower or "pos " in c_lower or "point of sale" in c_lower:
            m = re.sub(r"^(?:pos\s*(?:purchase|txn)?\s*(?:\d+)?\s*[-:]?\s*)", "", clean, flags=re.IGNORECASE)
            return TransactionType.BUY_GOODS, m.strip(), None

        # 6. Mobile Banking / Paybill / C2B / M-Pesa
        if "mpesa" in c_lower or "paybill" in c_lower or "c2b" in c_lower or "b2c" in c_lower:
            if "b2c" in c_lower or "received from mpesa" in c_lower or "c2b" in c_lower:
                return TransactionType.RECEIVE_MONEY, clean, None
            else:
                return TransactionType.PAYBILL, clean, None

        # 7. Standing Orders / Direct Debits
        if "standing order" in c_lower or "direct debit" in c_lower:
            beneficiary = re.sub(r"^(?:standing order|direct debit)\s*(?:to)?\s*[-:]?\s*", "", clean, flags=re.IGNORECASE)
            return TransactionType.PAYBILL, beneficiary.strip(), None

        # 8. ATM Withdrawals
        if "atm " in c_lower or "cash withdrawal" in c_lower or "atm wdl" in c_lower:
            agent = re.sub(r"^(?:atm\s*(?:cash|wdl|withdrawal)?\s*[-:]?\s*)", "", clean, flags=re.IGNORECASE)
            return TransactionType.WITHDRAWAL, f"ATM Cash Withdrawal {agent.strip()}", None

        # 9. Card online payments (Netflix, Uber, Amazon, etc.)
        if "visa" in c_lower or "mastercard" in c_lower:
            merchant = re.sub(r"^(?:visa|mastercard)\s*(?:debit|credit|purchase)?\s*[-:]?\s*", "", clean, flags=re.IGNORECASE)
            return TransactionType.BUY_GOODS, merchant.strip(), None

        # Default fallback
        return TransactionType.BUY_GOODS, clean, None

    def parse_pdf_stream(
        self, file_stream: io.BytesIO, password: Optional[str] = None
    ) -> List[ParsedTransaction]:
        """
        Reads Bank statement PDF pages, decrypts if required, and extracts transactions.
        Supports Equity Bank, KCB, Co-op, NCBA, Stanbic, Absa, DTB, I&M, etc.
        Handles both Date-in-middle (Equity format) and Date-at-start (KCB/Co-op format).
        """
        reader = PdfReader(file_stream)

        if reader.is_encrypted:
            if not password:
                raise ValueError("Bank statement PDF is password-protected. Please provide your password (usually your National ID or KRA PIN).")
            success = reader.decrypt(password)
            if not success:
                raise ValueError("Failed to decrypt Bank PDF. Incorrect password provided.")

        full_text_lines = []
        for page in reader.pages:
            text = None
            try:
                text = page.extract_text(extraction_mode="layout")
            except Exception:
                text = None
            if not text or len(text.strip()) < 50:
                try:
                    text = page.extract_text()
                except Exception:
                    text = ""
            if text:
                for line in text.splitlines():
                    cleaned = line.strip()
                    if cleaned:
                        full_text_lines.append(cleaned)

        transactions: List[ParsedTransaction] = []
        date_regex = re.compile(
            r"\b(?P<date>\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{4}[-/]\d{2}[-/]\d{2}|\d{1,2}-[A-Za-z]{3}-\d{2,4})\b"
        )
        amount_regex = re.compile(
            r"(?<!\w)(?:(?P<neg>-)?(?P<num>\d{1,3}(?:,\d{3})*\.\d{2}))(?!\w)"
        )

        header_skip = [
            "statement period", "statement date", "account created", "currency kes",
            "account number", "account branch", "customer name", "opening balance",
            "closing balance", "brought forward", "carried forward", "page "
        ]

        in_table = False
        accumulated_narrative: List[str] = []
        previous_balance: Optional[float] = None

        for idx, line in enumerate(full_text_lines):
            line_lower = line.lower()

            # Detect table boundary
            if any(tbl_kw in line_lower for tbl_kw in [
                "transaction details", "particulars", "narrative", "payment reference",
                "value date", "credit (money in)", "debit (money out)", "money in", "money out",
                "running balance", "transactions"
            ]):
                in_table = True
                accumulated_narrative = []
                continue

            # Skip metadata header lines outside the active transaction rows
            if any(skip_kw in line_lower for skip_kw in header_skip) and not any(k in line_lower for k in ["tpg", "pos", "app/mpesa"]):
                continue

            date_match = date_regex.search(line)
            amounts = amount_regex.findall(line)

            if date_match and amounts:
                d_str = date_match.group("date")
                num_floats = []
                for neg, num in amounts:
                    val = float(num.replace(",", ""))
                    num_floats.append(-val if neg == "-" else val)

                cleaned_line = line
                cleaned_line = date_regex.sub("", cleaned_line)
                cleaned_line = amount_regex.sub("", cleaned_line)
                cleaned_line = " ".join(cleaned_line.split())

                # Payment reference code extraction
                extracted_ref = None
                ref_match = re.search(r"\b([A-Z0-9]{8,14})\b", cleaned_line)
                if ref_match:
                    extracted_ref = ref_match.group(1)
                    cleaned_line = cleaned_line.replace(extracted_ref, "").strip()

                full_narrative = " ".join(accumulated_narrative + [cleaned_line]).strip()
                accumulated_narrative = []

                balance: Optional[float] = None
                tx_amount = 0.0
                is_credit = False

                if len(num_floats) >= 3:
                    # [Debit, Credit, Balance]
                    debit_val = abs(num_floats[0]) if num_floats[0] != 0 else 0.0
                    credit_val = abs(num_floats[1]) if num_floats[1] != 0 else 0.0
                    balance = num_floats[2]
                    if credit_val > 0 and debit_val == 0:
                        is_credit = True
                        tx_amount = credit_val
                    else:
                        is_credit = False
                        tx_amount = debit_val if debit_val > 0 else (credit_val or 0.0)
                elif len(num_floats) == 2:
                    raw_amt = abs(num_floats[0])
                    balance = num_floats[1]

                    # Mathematical delta resolution
                    if previous_balance is not None:
                        diff = round(balance - previous_balance, 2)
                        if diff > 0:
                            is_credit = True
                            tx_amount = diff
                        elif diff < 0:
                            is_credit = False
                            tx_amount = abs(diff)
                        else:
                            tx_amount = raw_amt
                            is_credit = False
                    else:
                        if any(in_kw in full_narrative.lower() for in_kw in [
                            "salary", "credit", "inward", "received", "refund", "deposit", "pesalink transfer"
                        ]):
                            is_credit = True
                        else:
                            is_credit = False
                        tx_amount = raw_amt
                else:
                    val = num_floats[0]
                    tx_amount = abs(val)
                    is_credit = (val > 0 and any(in_kw in full_narrative.lower() for in_kw in ["salary", "credit", "inward", "deposit"]))

                if balance is not None:
                    previous_balance = balance

                tx_type, counterparty, interp_ref = self._interpret_narrative(full_narrative)
                if interp_ref:
                    extracted_ref = interp_ref

                if is_credit:
                    if tx_type != TransactionType.RECEIVE_MONEY:
                        tx_type = TransactionType.RECEIVE_MONEY
                elif tx_type == TransactionType.RECEIVE_MONEY:
                    tx_type = TransactionType.BUY_GOODS

                timestamp = self._parse_date(d_str)
                tx_id = f"BNK{hashlib.md5((line + str(idx)).encode()).hexdigest()[:7].upper()}"
                masked_cp = self._mask_account_or_phone(counterparty)

                tx = ParsedTransaction(
                    transaction_id=tx_id,
                    tx_type=tx_type,
                    amount=tx_amount,
                    counterparty=counterparty,
                    counterparty_masked=masked_cp,
                    account_reference=extracted_ref,
                    timestamp=timestamp,
                    balance=balance,
                    tx_cost=0.0,
                    fuliza_amount=None,
                    raw_text=line,
                )
                transactions.append(tx)
            else:
                if in_table and not amounts:
                    accumulated_narrative.append(line)

        # Fallback 1: Text-based OCR via Gemini for non-standard or complex table layouts
        if not transactions:
            all_raw_text = "\n".join(full_text_lines)
            analyzer = GeminiFinancialAnalyzer()
            if len(all_raw_text.strip()) > 30:
                ocr_rows = analyzer.ocr_extract_table_from_text(all_raw_text)
                for idx, row in enumerate(ocr_rows):
                    try:
                        d_str = str(row.get("date", ""))
                        amt = abs(float(row.get("amount", 0.0)))
                        narr = str(row.get("narrative", "Bank Transaction")).strip()
                        is_cr = bool(row.get("is_credit", False))
                        bal = float(row.get("balance")) if row.get("balance") is not None else None

                        tx_type, cp, ref = self._interpret_narrative(narr)
                        if is_cr:
                            tx_type = TransactionType.RECEIVE_MONEY
                        elif tx_type == TransactionType.RECEIVE_MONEY:
                            tx_type = TransactionType.BUY_GOODS

                        tx = ParsedTransaction(
                            transaction_id=f"OCR{idx+1:04d}",
                            tx_type=tx_type,
                            amount=amt,
                            counterparty=cp,
                            counterparty_masked=self._mask_account_or_phone(cp),
                            account_reference=ref,
                            timestamp=self._parse_date(d_str),
                            balance=bal,
                            tx_cost=0.0,
                            fuliza_amount=None,
                            raw_text=narr,
                        )
                        transactions.append(tx)
                    except Exception:
                        continue

            # Fallback 2: Multimodal image-based OCR for scanned / image bank statements
            if not transactions:
                images_bytes = []
                for page in reader.pages:
                    try:
                        for img in getattr(page, "images", []):
                            images_bytes.append(img.data)
                    except Exception:
                        pass
                if images_bytes:
                    ocr_img_rows = analyzer.ocr_extract_from_page_images(images_bytes)
                    for idx, row in enumerate(ocr_img_rows):
                        try:
                            d_str = str(row.get("date", ""))
                            amt = abs(float(row.get("amount", 0.0)))
                            narr = str(row.get("narrative", "Bank Transaction")).strip()
                            is_cr = bool(row.get("is_credit", False))
                            bal = float(row.get("balance")) if row.get("balance") is not None else None

                            tx_type, cp, ref = self._interpret_narrative(narr)
                            if is_cr:
                                tx_type = TransactionType.RECEIVE_MONEY
                            elif tx_type == TransactionType.RECEIVE_MONEY:
                                tx_type = TransactionType.BUY_GOODS

                            tx = ParsedTransaction(
                                transaction_id=f"IMG{idx+1:04d}",
                                tx_type=tx_type,
                                amount=amt,
                                counterparty=cp,
                                counterparty_masked=self._mask_account_or_phone(cp),
                                account_reference=ref,
                                timestamp=self._parse_date(d_str),
                                balance=bal,
                                tx_cost=0.0,
                                fuliza_amount=None,
                                raw_text=narr,
                            )
                            transactions.append(tx)
                        except Exception:
                            continue

        return transactions

    def parse_csv_stream(self, file_content: str) -> List[ParsedTransaction]:
        """Parses CSV bank statement exports."""
        transactions: List[ParsedTransaction] = []
        reader = csv.reader(io.StringIO(file_content))
        rows = list(reader)

        if not rows:
            return transactions

        # Find header row
        header_idx = -1
        for idx, row in enumerate(rows[:10]):
            row_lower = [col.lower().strip() for col in row]
            if any("date" in c for c in row_lower) and any("desc" in c or "narrative" in c or "details" in c for c in row_lower):
                header_idx = idx
                break

        start_row = header_idx + 1 if header_idx != -1 else 0
        header = [c.lower().strip() for c in rows[header_idx]] if header_idx != -1 else []

        date_col = next((i for i, c in enumerate(header) if "date" in c), 0)
        desc_col = next((i for i, c in enumerate(header) if any(kw in c for kw in ["desc", "narrative", "details", "particulars"])), 1)
        debit_col = next((i for i, c in enumerate(header) if any(kw in c for kw in ["debit", "withdrawal", "paid out"])), None)
        credit_col = next((i for i, c in enumerate(header) if any(kw in c for kw in ["credit", "deposit", "paid in"])), None)
        bal_col = next((i for i, c in enumerate(header) if "balance" in c), None)

        for row_i, row in enumerate(rows[start_row:], start=start_row):
            if len(row) < 3:
                continue
            try:
                date_str = row[date_col].strip()
                if not re.search(r"\d", date_str):
                    continue
                narrative = row[desc_col].strip() if len(row) > desc_col else "Bank Transaction"

                debit = 0.0
                credit = 0.0
                if debit_col is not None and len(row) > debit_col and row[debit_col].strip():
                    try:
                        debit = abs(float(row[debit_col].replace(",", "").strip()))
                    except ValueError:
                        pass

                if credit_col is not None and len(row) > credit_col and row[credit_col].strip():
                    try:
                        credit = abs(float(row[credit_col].replace(",", "").strip()))
                    except ValueError:
                        pass

                bal = None
                if bal_col is not None and len(row) > bal_col and row[bal_col].strip():
                    try:
                        bal = float(row[bal_col].replace(",", "").strip())
                    except ValueError:
                        pass

                amount = credit if credit > 0 else debit
                tx_type, counterparty, acc_ref = self._interpret_narrative(narrative)
                if credit > 0:
                    tx_type = TransactionType.RECEIVE_MONEY

                timestamp = self._parse_date(date_str)
                tx_id = f"CSV{hashlib.md5((' '.join(row)).encode()).hexdigest()[:7].upper()}"

                tx = ParsedTransaction(
                    transaction_id=tx_id,
                    tx_type=tx_type,
                    amount=amount,
                    counterparty=counterparty,
                    counterparty_masked=self._mask_account_or_phone(counterparty),
                    account_reference=acc_ref,
                    timestamp=timestamp,
                    balance=bal,
                    tx_cost=0.0,
                    fuliza_amount=None,
                    raw_text=",".join(row),
                )
                transactions.append(tx)
            except Exception:
                continue

        return transactions

