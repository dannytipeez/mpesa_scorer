import os
import re
import base64
import time
from datetime import datetime
from typing import Dict, Optional, Tuple, Any
import requests
from dotenv import load_dotenv

# Load local environment if present
load_dotenv()


class MpesaGateway:
    def __init__(self):
        self.env = os.getenv("MPESA_ENVIRONMENT", "production").lower()
        self.consumer_key = os.getenv("MPESA_CONSUMER_KEY", "").strip()
        self.consumer_secret = os.getenv("MPESA_CONSUMER_SECRET", "").strip()
        self.passkey = os.getenv("MPESA_PASSKEY", "").strip()
        self.shortcode = os.getenv("MPESA_SHORTCODE", "5008412").strip()
        self.amount = int(os.getenv("MPESA_PAYMENT_AMOUNT", "50"))
        self.callback_url = os.getenv(
            "MPESA_CALLBACK_URL", "https://mpesa-scorer.onrender.com/api/v1/mpesa/callback"
        ).strip()
        self.transaction_type = os.getenv("MPESA_TRANSACTION_TYPE", "CustomerBuyGoodsOnline").strip()

        self.base_url = (
            "https://api.safaricom.co.ke"
            if self.env == "production"
            else "https://sandbox.safaricom.co.ke"
        )

        self._access_token: Optional[str] = None
        self._token_expiry_timestamp: float = 0.0

        # In-memory store for active checkout requests:
        # checkout_request_id -> { "status": "PENDING" | "COMPLETED" | "FAILED", ... }
        self.payments: Dict[str, Dict[str, Any]] = {}

    def format_phone_number(self, phone: str) -> str:
        """
        Normalizes Kenyan phone number to 254XXXXXXXXX format.
        Accepts: 0712345678, 0112345678, +254712345678, 254712345678, etc.
        """
        cleaned = re.sub(r"\D", "", phone)
        if cleaned.startswith("0") and len(cleaned) == 10:
            return "254" + cleaned[1:]
        elif cleaned.startswith("254") and len(cleaned) == 12:
            return cleaned
        elif len(cleaned) == 9 and (cleaned.startswith("7") or cleaned.startswith("1")):
            return "254" + cleaned
        raise ValueError(
            f"Invalid Kenyan phone number format: '{phone}'. Use 07XXXXXXXX or 01XXXXXXXX."
        )

    def get_access_token(self) -> str:
        """Retrieves and caches OAuth access token from Safaricom Daraja API."""
        now = time.time()
        if self._access_token and now < (self._token_expiry_timestamp - 60):
            return self._access_token

        if not self.consumer_key or not self.consumer_secret:
            raise ValueError(
                "M-Pesa Consumer Key or Consumer Secret is missing. Set them in environment variables."
            )

        auth_url = f"{self.base_url}/oauth/v1/generate?grant_type=client_credentials"
        auth_str = base64.b64encode(
            f"{self.consumer_key}:{self.consumer_secret}".encode("utf-8")
        ).decode("utf-8")

        headers = {
            "Authorization": f"Basic {auth_str}",
            "Content-Type": "application/json"
        }

        resp = requests.get(auth_url, headers=headers, timeout=12)
        if resp.status_code != 200:
            raise RuntimeError(
                f"Daraja OAuth authentication failed ({resp.status_code}): {resp.text}"
            )

        data = resp.json()
        token = data.get("access_token")
        expires_in = int(data.get("expires_in", 3599))

        self._access_token = token
        self._token_expiry_timestamp = now + expires_in
        return token

    def initiate_stk_push(
        self,
        phone: str,
        amount: Optional[int] = None,
        reference: str = "AfriqanScore"
    ) -> Dict[str, Any]:
        """
        Initiates an M-Pesa STK Push to the customer's phone for 50 KES.
        """
        formatted_phone = self.format_phone_number(phone)
        charge_amount = amount or self.amount
        token = self.get_access_token()

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        password_str = f"{self.shortcode}{self.passkey}{timestamp}"
        password = base64.b64encode(password_str.encode("utf-8")).decode("utf-8")

        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }

        # Try configured transaction type first; fallback if Till/Paybill mismatch occurs
        tx_types_to_try = [self.transaction_type]
        if self.transaction_type == "CustomerBuyGoodsOnline":
            tx_types_to_try.append("CustomerPayBillOnline")
        else:
            tx_types_to_try.append("CustomerBuyGoodsOnline")

        last_error = None
        for tx_type in tx_types_to_try:
            payload = {
                "BusinessShortCode": self.shortcode,
                "Password": password,
                "Timestamp": timestamp,
                "TransactionType": tx_type,
                "Amount": charge_amount,
                "PartyA": formatted_phone,
                "PartyB": self.shortcode,
                "PhoneNumber": formatted_phone,
                "CallBackURL": self.callback_url,
                "AccountReference": reference[:12],
                "TransactionDesc": "Afriqan Credit Report"
            }

            url = f"{self.base_url}/mpesa/stkpush/v1/processrequest"
            resp = requests.post(url, json=payload, headers=headers, timeout=15)
            data = resp.json()

            if resp.status_code == 200 and data.get("ResponseCode") == "0":
                checkout_id = data.get("CheckoutRequestID")
                merchant_id = data.get("MerchantRequestID")
                desc = data.get("ResponseDescription", "Success. Request accepted for processing")

                # Track in-memory
                self.payments[checkout_id] = {
                    "checkout_request_id": checkout_id,
                    "merchant_request_id": merchant_id,
                    "phone": formatted_phone,
                    "amount": charge_amount,
                    "status": "PENDING",
                    "created_at": time.time(),
                    "mpesa_receipt": None,
                    "result_desc": desc
                }

                return {
                    "success": True,
                    "checkout_request_id": checkout_id,
                    "merchant_request_id": merchant_id,
                    "customer_message": data.get("CustomerMessage", "STK push sent to your phone.")
                }
            else:
                last_error = data.get("errorMessage") or data.get("ResponseDescription") or str(data)

        raise RuntimeError(f"STK Push failed from Safaricom: {last_error}")

    def process_callback(self, callback_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles Safaricom asynchronous callback payload.
        """
        body = callback_data.get("Body", {})
        stk_cb = body.get("stkCallback", {})
        checkout_id = stk_cb.get("CheckoutRequestID")
        result_code = stk_cb.get("ResultCode")
        result_desc = stk_cb.get("ResultDesc", "")

        if not checkout_id:
            return {"status": "ignored", "reason": "Missing CheckoutRequestID"}

        meta_items = stk_cb.get("CallbackMetadata", {}).get("Item", [])
        meta_dict = {item.get("Name"): item.get("Value") for item in meta_items if "Name" in item}

        receipt = meta_dict.get("MpesaReceiptNumber")
        amount = meta_dict.get("Amount")
        phone = meta_dict.get("PhoneNumber")

        if result_code == 0:
            status = "COMPLETED"
        else:
            status = "FAILED"

        record = self.payments.get(checkout_id, {})
        record.update({
            "checkout_request_id": checkout_id,
            "status": status,
            "result_code": result_code,
            "result_desc": result_desc,
            "mpesa_receipt": receipt,
            "amount": amount or record.get("amount", self.amount),
            "phone": str(phone) if phone else record.get("phone"),
            "updated_at": time.time()
        })
        self.payments[checkout_id] = record

        return {
            "checkout_request_id": checkout_id,
            "status": status,
            "mpesa_receipt": receipt
        }

    def query_status(self, checkout_request_id: str) -> Dict[str, Any]:
        """
        Checks payment status locally, and if still PENDING, queries Safaricom STK Push Query API.
        """
        record = self.payments.get(checkout_request_id)
        if not record:
            return {"status": "NOT_FOUND", "message": "Unknown transaction identifier."}

        # If already completed or failed, return immediately
        if record["status"] in ("COMPLETED", "FAILED"):
            return record

        # If pending for less than 3 seconds, let callback or polling arrive
        # If still pending, query Safaricom
        try:
            token = self.get_access_token()
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            password_str = f"{self.shortcode}{self.passkey}{timestamp}"
            password = base64.b64encode(password_str.encode("utf-8")).decode("utf-8")

            url = f"{self.base_url}/mpesa/stkpushquery/v1/query"
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json"
            }
            payload = {
                "BusinessShortCode": self.shortcode,
                "Password": password,
                "Timestamp": timestamp,
                "CheckoutRequestID": checkout_request_id
            }

            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            data = resp.json()
            res_code = str(data.get("ResultCode", ""))

            if res_code == "0":
                record["status"] = "COMPLETED"
                record["result_desc"] = data.get("ResultDesc", "Transaction successful.")
            elif res_code in ("1032", "1037", "1", "2001"):
                record["status"] = "FAILED"
                record["result_desc"] = data.get("ResultDesc", "Payment not completed.")
        except Exception:
            pass

        return record

    def simulate_mock_approval(self, checkout_request_id: str) -> Dict[str, Any]:
        """
        Helper for testing or demo purposes when needed.
        """
        record = self.payments.get(checkout_request_id, {
            "checkout_request_id": checkout_request_id,
            "amount": 50,
            "phone": "254700000000"
        })
        record["status"] = "COMPLETED"
        record["mpesa_receipt"] = f"DEMO{int(time.time())}"
        record["result_desc"] = "Demo Payment Approved"
        self.payments[checkout_request_id] = record
        return record

