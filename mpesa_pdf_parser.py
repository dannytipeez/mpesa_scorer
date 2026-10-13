import io
import re
from datetime import datetime
from typing import List, Optional, Tuple
from pypdf import PdfReader
from mpesa_parser import MpesaParser, ParsedTransaction, TransactionType


class MpesaPdfParser:
    """
    Parses official Safaricom M-Pesa PDF e-statements into structured ParsedTransaction models.
    Supports password-protected PDFs (National ID / Passport number) and standard multi-page statements.
    """

    def __init__(self, salt: str = "mpesa_local_salt_123"):
        self.salt = salt
        self.sms_parser = MpesaParser(salt=salt)

        # Regex for receipt start: e.g. "SH2819DK3L 2026-03-12 16:15:00 ..." or "12/03/2026 16:15:00"
        self.tx_line_regex = re.compile(
            r"^(?P<receipt>[A-Z0-9]{10})\s+"
            r"(?P<datetime>\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}|\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}|\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2})\s+"
            r"(?P<rest>.+)$",
            re.IGNORECASE,
        )

        # Regex to match the trailing amounts: [Paid In] [Withdrawn] [Balance]
        # Examples: "Completed 60,000.00 72,000.00" or "Completed - 1,500.00 4,210.00" or "Completed -1,500.00 4,210.00"
        self.amounts_regex = re.compile(
            r"(?:Completed|Failed|Reversed)?\s*"
            r"(?:(?P<paid_in>[\d,]+\.\d{2})|-)?\s*"
            r"(?:(?P<withdrawn>[\d,]+\.\d{2}|-[\d,]+\.\d{2})|-)?\s*"
            r"(?P<balance>-?[\d,]+\.\d{2})$",
            re.IGNORECASE,
        )

    def _parse_timestamp(self, dt_str: str) -> datetime:
        dt_clean = " ".join(dt_str.strip().split())
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%d/%m/%Y %H:%M:%S",
            "%d-%m-%Y %H:%M:%S",
            "%d/%m/%y %H:%M:%S",
            "%Y-%m-%d %I:%M %p",
            "%d/%m/%Y %I:%M %p",
        ]
        for fmt in formats:
            try:
                return datetime.strptime(dt_clean, fmt)
            except ValueError:
                pass
        return datetime.now()

    def _interpret_details(
        self, details_raw: str
    ) -> Tuple[TransactionType, str, Optional[str], Optional[float]]:
        """
        Extracts tx_type, counterparty, account_ref, and fuliza_amount from Details column text.
        """
        details = " ".join(details_raw.strip().split())
        d_lower = details.lower()
        fuliza_amount = None

        # Check Fuliza mentions
        fuliza_match = re.search(r"fuliza.*?ksh\s*([\d,]+\.?\d*)", details, re.IGNORECASE)
        if fuliza_match:
            try:
                fuliza_amount = float(fuliza_match.group(1).replace(",", ""))
            except ValueError:
                pass

        # 1. Pay Bill: e.g. "Pay Bill to 888880 - KPLC PREPAID Acc. 482910394"
        if "pay bill" in d_lower or "paybill" in d_lower:
            pb_match = re.search(
                r"pay\s*bill\s*(?:to\s*)?(?:\d+\s*-\s*)?(?P<biz>.+?)(?:\s+acc(?:ount|\.)?\s+(?P<acc>[^\s]+))?$",
                details,
                re.IGNORECASE,
            )
            if pb_match:
                biz = pb_match.group("biz").strip()
                acc = pb_match.group("acc").strip() if pb_match.group("acc") else None
                return TransactionType.PAYBILL, biz, acc, fuliza_amount
            return TransactionType.PAYBILL, details, None, fuliza_amount

        # 2. Buy Goods / Merchant Payment: e.g. "Merchant Payment to 28491 - QUICKMART TOM MBOYA"
        if "merchant payment" in d_lower or "buy goods" in d_lower:
            bg_match = re.search(
                r"(?:merchant payment|buy goods)\s*(?:to\s*)?(?:\d+\s*-\s*)?(?P<merchant>.+)$",
                details,
                re.IGNORECASE,
            )
            if bg_match:
                merchant = bg_match.group("merchant").strip()
                return TransactionType.BUY_GOODS, merchant, None, fuliza_amount
            return TransactionType.BUY_GOODS, details, None, fuliza_amount

        # 3. Customer Transfer (P2P Send): e.g. "Customer Transfer to 254722112233 - JOHN KAMAU"
        if "customer transfer to" in d_lower or "transfer to" in d_lower:
            send_match = re.search(
                r"(?:customer transfer|transfer)\s+to\s+(?P<recipient>.+)$",
                details,
                re.IGNORECASE,
            )
            if send_match:
                recipient = send_match.group("recipient").strip()
                return TransactionType.SEND_MONEY, recipient, None, fuliza_amount
            return TransactionType.SEND_MONEY, details, None, fuliza_amount

        # 4. Funds Received (P2P Receive): e.g. "Funds received from 2547... ACME TECH SOLUTIONS"
        if "funds received" in d_lower or "received from" in d_lower:
            rec_match = re.search(
                r"(?:funds received from|received from)\s+(?P<sender>.+)$",
                details,
                re.IGNORECASE,
            )
            if rec_match:
                sender = rec_match.group("sender").strip()
                return TransactionType.RECEIVE_MONEY, sender, None, fuliza_amount
            return TransactionType.RECEIVE_MONEY, details, None, fuliza_amount

        # 5. Agent / ATM Withdrawal: e.g. "Withdrawal from Agent 12345 - AGENT STORE"
        if "withdrawal" in d_lower or "withdraw" in d_lower:
            w_match = re.search(
                r"(?:withdrawal from agent|withdraw from)\s*(?:\d+\s*-\s*)?(?P<agent>.+)$",
                details,
                re.IGNORECASE,
            )
            if w_match:
                agent = w_match.group("agent").strip()
                return TransactionType.WITHDRAWAL, agent, None, fuliza_amount
            return TransactionType.WITHDRAWAL, details, None, fuliza_amount

        # 6. Airtime Purchase
        if "airtime" in d_lower:
            return TransactionType.AIRTIME, "Safaricom Airtime", None, fuliza_amount

        # 7. Fuliza Overdraft standalone event
        if "overdraft" in d_lower and "fuliza" in d_lower:
            return TransactionType.RECEIVE_MONEY, "Fuliza M-PESA Overdraft", None, fuliza_amount

        return TransactionType.UNKNOWN, details, None, fuliza_amount

    def parse_pdf_stream(
        self, file_stream: io.BytesIO, password: Optional[str] = None
    ) -> List[ParsedTransaction]:
        """Reads PDF pages, decrypts if needed, and extracts M-Pesa transactions."""
        reader = PdfReader(file_stream)

        if reader.is_encrypted:
            if not password:
                raise ValueError("PDF is password protected. Please provide your M-Pesa statement password (usually your National ID / Passport number).")
            success = reader.decrypt(password)
            if not success:
                raise ValueError("Failed to decrypt PDF. Incorrect password provided.")

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
                receipt = match.group("receipt")
                dt_str = match.group("datetime")
                rest = match.group("rest").strip()

                # Some transactions wrap details across 1-2 lines before amounts
                combined_rest = rest
                next_idx = i + 1
                while next_idx < n:
                    next_line = full_text_lines[next_idx]
                    # If next line is a new transaction receipt or page header, break
                    if self.tx_line_regex.match(next_line) or "M-PESA STATEMENT" in next_line:
                        break
                    # If current combined_rest already has trailing balance amount, break
                    if re.search(r"-?[\d,]+\.\d{2}$", combined_rest):
                        break
                    combined_rest += " " + next_line
                    next_idx += 1
                i = next_idx - 1

                # Parse amounts from combined_rest
                # Usually: Details ... [Status] [Paid In] [Withdrawn] [Balance]
                amounts_match = self.amounts_regex.search(combined_rest)
                if amounts_match:
                    details_chunk = combined_rest[: amounts_match.start()].strip()
                    paid_in_str = amounts_match.group("paid_in")
                    withdrawn_str = amounts_match.group("withdrawn")
                    balance_str = amounts_match.group("balance")

                    # Calculate amount and determine inflow vs outflow
                    amount = 0.0
                    paid_in = float(paid_in_str.replace(",", "")) if paid_in_str and paid_in_str != "-" else 0.0
                    withdrawn = (
                        abs(float(withdrawn_str.replace(",", "")))
                        if withdrawn_str and withdrawn_str != "-"
                        else 0.0
                    )

                    balance = float(balance_str.replace(",", "")) if balance_str else None

                    tx_type, counterparty, acc_ref, fuliza = self._interpret_details(details_chunk)

                    if paid_in > 0:
                        amount = paid_in
                        if tx_type == TransactionType.UNKNOWN:
                            tx_type = TransactionType.RECEIVE_MONEY
                    elif withdrawn > 0:
                        amount = withdrawn
                        if tx_type == TransactionType.UNKNOWN:
                            tx_type = TransactionType.PAYBILL

                    timestamp = self._parse_timestamp(dt_str)

                    # Redact PII in counterparty
                    masked_counterparty = self.sms_parser._mask_phone(counterparty)

                    tx = ParsedTransaction(
                        transaction_id=receipt,
                        tx_type=tx_type,
                        amount=amount,
                        counterparty=counterparty,
                        counterparty_masked=masked_counterparty,
                        account_reference=acc_ref,
                        timestamp=timestamp,
                        balance=balance,
                        tx_cost=0.0,
                        fuliza_amount=fuliza,
                        raw_text=line,
                    )
                    transactions.append(tx)
            i += 1

        return transactions

    def parse_pdf_file(self, file_path: str, password: Optional[str] = None) -> List[ParsedTransaction]:
        with open(file_path, "rb") as f:
            return self.parse_pdf_stream(io.BytesIO(f.read()), password=password)

