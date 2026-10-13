import csv
import hashlib
import io
import re
from datetime import datetime
from typing import List, Optional, Tuple
from pypdf import PdfReader
from mpesa_parser import ParsedTransaction, TransactionType


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
        acc_ref = None

        # Clean common bank transaction prefixes
        # 1. POS Purchases (Cards at merchants / till)
        if "pos purchase" in c_lower or "pos " in c_lower or "point of sale" in c_lower:
            m = re.sub(r"^(?:pos\s*(?:purchase|txn)?\s*(?:\d+)?\s*[-:]?\s*)", "", clean, flags=re.IGNORECASE)
            return TransactionType.BUY_GOODS, m.strip(), None

        # 2. PesaLink / EFT / RTGS Outward or Inward
        if any(kw in c_lower for kw in ["pesalink", "eft", "rtgs"]):
            if any(in_kw in c_lower for in_kw in ["received", "inward", "credit", "cr to", "salary"]):
                sender = re.sub(r"^(?:pesalink|eft|rtgs)\s*(?:inward|credit|received from)?\s*[-:]?\s*", "", clean, flags=re.IGNORECASE)
                return TransactionType.RECEIVE_MONEY, sender.strip(), None
            else:
                recipient = re.sub(r"^(?:pesalink|eft|rtgs)\s*(?:outward|transfer to|payment to)?\s*[-:]?\s*", "", clean, flags=re.IGNORECASE)
                return TransactionType.SEND_MONEY, recipient.strip(), None

        # 3. Mobile Banking / Paybill / C2B / M-Pesa to Bank
        if "mpesa" in c_lower or "paybill" in c_lower or "c2b" in c_lower or "b2c" in c_lower:
            if "b2c" in c_lower or "received from mpesa" in c_lower or "c2b" in c_lower:
                return TransactionType.RECEIVE_MONEY, clean, None
            else:
                return TransactionType.PAYBILL, clean, None

        # 4. Standing Orders / Direct Debits
        if "standing order" in c_lower or "direct debit" in c_lower:
            beneficiary = re.sub(r"^(?:standing order|direct debit)\s*(?:to)?\s*[-:]?\s*", "", clean, flags=re.IGNORECASE)
            return TransactionType.PAYBILL, beneficiary.strip(), None

        # 5. ATM Withdrawals
        if "atm " in c_lower or "cash withdrawal" in c_lower or "atm wdl" in c_lower:
            agent = re.sub(r"^(?:atm\s*(?:cash|wdl|withdrawal)?\s*[-:]?\s*)", "", clean, flags=re.IGNORECASE)
            return TransactionType.WITHDRAWAL, f"ATM Cash Withdrawal {agent.strip()}", None

        # 6. Bank Charges / Ledger fees / Excise duty
        if any(chg in c_lower for chg in ["ledger fee", "excise duty", "maintenance fee", "alert charge", "sms fee"]):
            return TransactionType.PAYBILL, f"Bank Charges: {clean}", None

        # 7. Card online payments (Netflix, Uber, Amazon, etc.)
        if "visa" in c_lower or "mastercard" in c_lower:
            merchant = re.sub(r"^(?:visa|mastercard)\s*(?:debit|credit|purchase)?\s*[-:]?\s*", "", clean, flags=re.IGNORECASE)
            return TransactionType.BUY_GOODS, merchant.strip(), None

        # Default fallback
        return TransactionType.BUY_GOODS, clean, None

    def parse_pdf_stream(
        self, file_stream: io.BytesIO, password: Optional[str] = None
    ) -> List[ParsedTransaction]:
        """Reads Bank statement PDF pages, decrypts if required, and extracts transactions."""
        reader = PdfReader(file_stream)

        if reader.is_encrypted:
            if not password:
                raise ValueError("Bank statement PDF is password-protected. Please provide your password (usually your National ID or KRA PIN).")
            success = reader.decrypt(password)
            if not success:
                raise ValueError("Failed to decrypt Bank PDF. Incorrect password provided.")

        full_text_lines = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                for line in text.splitlines():
                    cleaned = line.strip()
                    if cleaned:
                        full_text_lines.append(cleaned)

        transactions: List[ParsedTransaction] = []
        i = 0
        n = len(full_text_lines)

        while i < n:
            line = full_text_lines[i]
            match = self.tx_line_regex.match(line)

            if match:
                date_str = match.group("date")
                rest = match.group("rest").strip()

                # Handle multiline narrative wrapping
                combined_rest = rest
                next_idx = i + 1
                while next_idx < n:
                    next_line = full_text_lines[next_idx]
                    if self.tx_line_regex.match(next_line) or "ACCOUNT STATEMENT" in next_line:
                        break
                    if re.search(r"-?[\d,]+\.\d{2}$", combined_rest):
                        break
                    combined_rest += " " + next_line
                    next_idx += 1
                i = next_idx - 1

                # Extract trailing amounts: [Debit] [Credit] [Balance]
                amounts_match = self.amounts_regex.search(combined_rest)
                if amounts_match:
                    narrative = combined_rest[: amounts_match.start()].strip()
                    amt1 = amounts_match.group("amount1")
                    amt2 = amounts_match.group("amount2")
                    balance_str = amounts_match.group("balance")

                    # Deduce debit vs credit
                    # Usually: Debit is amt1, Credit is amt2 (or vice versa depending on column alignment)
                    debit = 0.0
                    credit = 0.0

                    if amt2 and amt2 != "-":
                        # Both columns present: amt1 is Debit, amt2 is Credit
                        try:
                            debit = float(amt1.replace(",", "")) if amt1 != "-" else 0.0
                        except ValueError:
                            pass
                        try:
                            credit = float(amt2.replace(",", ""))
                        except ValueError:
                            pass
                    else:
                        # Single amount column: check if negative or narrative context
                        try:
                            val = float(amt1.replace(",", ""))
                            if val < 0 or any(d_kw in narrative.lower() for d_kw in ["pos", "withdrawal", "wdl", "debit", "fee"]):
                                debit = abs(val)
                            else:
                                credit = val
                        except ValueError:
                            pass

                    balance = float(balance_str.replace(",", "")) if balance_str else None
                    tx_type, counterparty, acc_ref = self._interpret_narrative(narrative)

                    if credit > 0:
                        amount = credit
                        if tx_type != TransactionType.RECEIVE_MONEY:
                            tx_type = TransactionType.RECEIVE_MONEY
                    else:
                        amount = debit if debit > 0 else 0.0

                    timestamp = self._parse_date(date_str)
                    tx_id = f"BNK{hashlib.md5((line + str(i)).encode()).hexdigest()[:7].upper()}"
                    masked_cp = self._mask_account_or_phone(counterparty)

                    tx = ParsedTransaction(
                        transaction_id=tx_id,
                        tx_type=tx_type,
                        amount=amount,
                        counterparty=counterparty,
                        counterparty_masked=masked_cp,
                        account_reference=acc_ref,
                        timestamp=timestamp,
                        balance=balance,
                        tx_cost=0.0,
                        fuliza_amount=None,
                        raw_text=line,
                    )
                    transactions.append(tx)
            i += 1

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

