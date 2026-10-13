import hashlib
import re
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple
from pydantic import BaseModel, Field


class TransactionType(str, Enum):
    SEND_MONEY = "SEND_MONEY"
    RECEIVE_MONEY = "RECEIVE_MONEY"
    PAYBILL = "PAYBILL"
    BUY_GOODS = "BUY_GOODS"
    WITHDRAWAL = "WITHDRAWAL"
    AIRTIME = "AIRTIME"
    UNKNOWN = "UNKNOWN"


class ParsedTransaction(BaseModel):
    transaction_id: str
    tx_type: TransactionType
    amount: float = Field(..., ge=0.0)
    counterparty: str
    counterparty_masked: str
    account_reference: Optional[str] = None
    timestamp: datetime
    balance: Optional[float] = None
    tx_cost: float = 0.0
    fuliza_amount: Optional[float] = None
    raw_text: str


class MpesaParser:
    def __init__(self, salt: str = "mpesa_local_salt_123"):
        self.salt = salt
        self.dt_regex = (
            r"on\s+(?P<date>\d{1,2}/\d{1,2}/\d{2,4})\s+at\s+(?P<time>\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))"
        )

        # 1. Send Money (P2P)
        self.send_regex = re.compile(
            r"^(?P<id>[A-Z0-9]+)\s+Confirmed\.\s+Ksh(?P<amount>[\d,]+\.\d{2})\s+sent to\s+(?P<recipient>.+?)\s+"
            + self.dt_regex,
            re.IGNORECASE,
        )

        # 2. Receive Money
        self.receive_regex = re.compile(
            r"^(?P<id>[A-Z0-9]+)\s+Confirmed\.\s+You have received\s+Ksh(?P<amount>[\d,]+\.\d{2})\s+from\s+(?P<sender>.+?)\s+"
            + self.dt_regex,
            re.IGNORECASE,
        )

        # 3. Paybill (Pay to business with account)
        self.paybill_regex = re.compile(
            r"^(?P<id>[A-Z0-9]+)\s+Confirmed\.\s+Ksh(?P<amount>[\d,]+\.\d{2})\s+paid to\s+(?P<biz>.+?)\.?\s+for account\s+(?P<acc>[^\.]+)\.?\s+"
            + self.dt_regex,
            re.IGNORECASE,
        )

        # 4. Buy Goods (Till Number)
        self.buy_goods_regex = re.compile(
            r"^(?P<id>[A-Z0-9]+)\s+Confirmed\.\s+Ksh(?P<amount>[\d,]+\.\d{2})\s+paid to\s+(?P<merchant>.+?)\.?\s+"
            + self.dt_regex,
            re.IGNORECASE,
        )

        # 5. Agent / ATM Withdrawal
        self.withdraw_regex = re.compile(
            r"^(?P<id>[A-Z0-9]+)\s+Confirmed\.\s+Withdraw\s+Ksh(?P<amount>[\d,]+\.\d{2})\s+from\s+(?P<agent>.+?)\s+"
            + self.dt_regex,
            re.IGNORECASE,
        )

        # 6. Airtime Purchase
        self.airtime_regex = re.compile(
            r"^(?P<id>[A-Z0-9]+)\s+Confirmed\.\s+You bought\s+Ksh(?P<amount>[\d,]+\.\d{2})\s+of airtime\s+"
            + self.dt_regex,
            re.IGNORECASE,
        )

    def _mask_phone(self, text: str) -> str:
        """Finds phone numbers (07xx/01xx/+254xx) and hashes them with salt."""
        def replace_fn(match):
            phone = match.group(0)
            hashed = hashlib.sha256((phone + self.salt).encode()).hexdigest()[:10]
            return f"[PHONE_ID_{hashed}]"

        return re.sub(r"(?:\+?254|0)[17]\d{8}", replace_fn, text)

    def _parse_meta(self, text: str) -> Tuple[Optional[float], float, Optional[float]]:
        # Balance
        bal_match = re.search(r"New M-PESA balance is Ksh(?P<bal>[\d,\.\-]+)", text, re.IGNORECASE)
        balance = None
        if bal_match:
            try:
                balance = float(bal_match.group("bal").rstrip(".").replace(",", ""))
            except ValueError:
                pass

        # Transaction Cost
        cost_match = re.search(r"Transaction cost,?\s+Ksh(?P<cost>[\d,]+\.\d{2})", text, re.IGNORECASE)
        cost = float(cost_match.group("cost").replace(",", "")) if cost_match else 0.0

        # Fuliza Overdraft
        fuliza_match = re.search(r"Fuliza M-PESA amount is Ksh(?P<fuliza>[\d,]+\.\d{2})", text, re.IGNORECASE)
        fuliza = float(fuliza_match.group("fuliza").replace(",", "")) if fuliza_match else None

        return balance, cost, fuliza

    def _parse_timestamp(self, date_str: str, time_str: str) -> datetime:
        d = date_str.strip()
        t = re.sub(r"\s+", " ", time_str.strip().upper())
        # Handles 2-digit vs 4-digit years
        year_part = d.split("/")[-1]
        fmt = "%d/%m/%y %I:%M %p" if len(year_part) == 2 else "%d/%m/%Y %I:%M %p"
        return datetime.strptime(f"{d} {t}", fmt)

    def parse(self, text: str) -> Optional[ParsedTransaction]:
        text_clean = " ".join(text.strip().split())
        balance, cost, fuliza = self._parse_meta(text_clean)

        # 1. Paybill check (prioritized because it may share words with Buy Goods)
        if "for account" in text_clean.lower():
            if match := self.paybill_regex.search(text_clean):
                return ParsedTransaction(
                    transaction_id=match.group("id"),
                    tx_type=TransactionType.PAYBILL,
                    amount=float(match.group("amount").replace(",", "")),
                    counterparty=match.group("biz").strip(),
                    counterparty_masked=self._mask_phone(match.group("biz").strip()),
                    account_reference=match.group("acc").strip(),
                    timestamp=self._parse_timestamp(match.group("date"), match.group("time")),
                    balance=balance,
                    tx_cost=cost,
                    fuliza_amount=fuliza,
                    raw_text=text,
                )

        # 2. P2P Send
        if match := self.send_regex.search(text_clean):
            return ParsedTransaction(
                transaction_id=match.group("id"),
                tx_type=TransactionType.SEND_MONEY,
                amount=float(match.group("amount").replace(",", "")),
                counterparty=match.group("recipient").strip(),
                counterparty_masked=self._mask_phone(match.group("recipient").strip()),
                timestamp=self._parse_timestamp(match.group("date"), match.group("time")),
                balance=balance,
                tx_cost=cost,
                fuliza_amount=fuliza,
                raw_text=text,
            )

        # 3. Receive Money
        if match := self.receive_regex.search(text_clean):
            return ParsedTransaction(
                transaction_id=match.group("id"),
                tx_type=TransactionType.RECEIVE_MONEY,
                amount=float(match.group("amount").replace(",", "")),
                counterparty=match.group("sender").strip(),
                counterparty_masked=self._mask_phone(match.group("sender").strip()),
                timestamp=self._parse_timestamp(match.group("date"), match.group("time")),
                balance=balance,
                tx_cost=cost,
                fuliza_amount=fuliza,
                raw_text=text,
            )

        # 4. Buy Goods (Till)
        if match := self.buy_goods_regex.search(text_clean):
            return ParsedTransaction(
                transaction_id=match.group("id"),
                tx_type=TransactionType.BUY_GOODS,
                amount=float(match.group("amount").replace(",", "")),
                counterparty=match.group("merchant").strip(),
                counterparty_masked=self._mask_phone(match.group("merchant").strip()),
                timestamp=self._parse_timestamp(match.group("date"), match.group("time")),
                balance=balance,
                tx_cost=cost,
                fuliza_amount=fuliza,
                raw_text=text,
            )

        # 5. Withdrawal
        if match := self.withdraw_regex.search(text_clean):
            return ParsedTransaction(
                transaction_id=match.group("id"),
                tx_type=TransactionType.WITHDRAWAL,
                amount=float(match.group("amount").replace(",", "")),
                counterparty=match.group("agent").strip(),
                counterparty_masked=self._mask_phone(match.group("agent").strip()),
                timestamp=self._parse_timestamp(match.group("date"), match.group("time")),
                balance=balance,
                tx_cost=cost,
                fuliza_amount=fuliza,
                raw_text=text,
            )

        # 6. Airtime
        if match := self.airtime_regex.search(text_clean):
            return ParsedTransaction(
                transaction_id=match.group("id"),
                tx_type=TransactionType.AIRTIME,
                amount=float(match.group("amount").replace(",", "")),
                counterparty="Safaricom Airtime",
                counterparty_masked="Safaricom Airtime",
                timestamp=self._parse_timestamp(match.group("date"), match.group("time")),
                balance=balance,
                tx_cost=cost,
                fuliza_amount=fuliza,
                raw_text=text,
            )

        return None

