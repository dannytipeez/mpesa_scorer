import random
import time
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def get_vocabularies():
    """Returns rich, realistic Kenyan entities, brands, and personal names."""
    locations = [
        "Westlands", "Kilimani", "Lavington", "Karen", "CBD Nairobi", "Parklands",
        "Roysambu", "Langata", "Ngong Road", "Thika Road", "Mombasa Road", "Buruburu",
        "South C", "South B", "Kasarani", "Ruiru", "Kiambu Town", "Nakuru Town",
        "Mombasa Nyali", "Mombasa Diani", "Kisumu Milimani", "Eldoret Town",
        "Kitengela", "Ongata Rongai", "Utawala", "Embakasi", "Donholm", "Githurai",
        "Kikuyu", "Naivasha", "Nanyuki", "Machakos", "Meru", "Kakamega", "Kericho",
        "Eastleigh", "Syokimau", "Kahawa Sukari", "Kahawa West", "Upper Hill", "Kileleshwa",
        "Loresho", "Hurlingham", "Muthaiga", "Gigiri", "Runda", "Ruaka", "Juja"
    ]

    first_names = [
        "John", "Mary", "Peter", "Jane", "James", "Grace", "David", "Faith", "Joseph",
        "Mercy", "Daniel", "Esther", "Samuel", "Joy", "Paul", "Brian", "Kevin", "Denis",
        "Evans", "Victor", "Lucy", "Beatrice", "Rose", "Christine", "Sarah", "Mohammed",
        "Fatuma", "Hassan", "Amina", "Ali", "Ahmed", "Hussein", "Abdullahi", "Omar",
        "Collins", "Geoffrey", "Caroline", "Jackline", "Sharon", "Kennedy", "Kelvin",
        "Emmanuel", "Francis", "Stephen", "Martin", "Patrick", "Purity", "Brenda", "George"
    ]
    last_names = [
        "Kamau", "Wanjiku", "Njoroge", "Wangari", "Kariuki", "Mwangi", "Maina", "Kimani",
        "Nyambura", "Ochieng", "Odhiambo", "Otieno", "Anyango", "Atieno", "Omondi",
        "Kiprop", "Cheruiyot", "Kiprono", "Chebet", "Rotich", "Korir", "Chemutai",
        "Mutua", "Musyoka", "Mutiso", "Mwende", "Kilonzo", "Wamalwa", "Simiyu", "Wafula",
        "Nekesa", "Barasa", "Juma", "Karanja", "Muthoni", "Gacheru", "Ndung'u", "Ouma",
        "Wekesa", "Moraa", "Mogaka", "Makori", "Onyango", "Achieng", "Koech", "Kibet"
    ]

    seeds = {
        "Food & Dining": [
            "Java House", "Artcaffe", "KFC", "Dominos Pizza", "CJ Restaurant", "CJ's",
            "Big Square", "Galitos", "Pizza Inn", "Chicken Inn", "Creamy Inn", "Subway",
            "Steers", "Debonairs Pizza", "Planet Yogurt", "Mama Otieno Fish Point",
            "Kilimanjaro Restaurant", "Swahili Dishes", "City Nyama Choma", "Charcoal Grill",
            "Broadway Bakery Delight", "Campus Canteen", "Tacos and Burgers", "Coffee Point",
            "Urban Eatery", "Kempinski Cafe", "Sarova Restaurant", "Intercontinental Diner",
            "Nairobi Kitchen", "Pizza Mojo", "Kuku Foods", "Nyama Bite", "Roast House",
            "Sweet Treats Bakery", "Bake House", "Choma Zone", "Fish Grill", "Shawarma Hub",
            "Mama Nilishe Eatery", "Wimpy Fast Food", "Toranj Persian Restaurant", "Habesha",
            "Fogo Gaucho Brazilian", "Talisman Restaurant", "About Thyme", "Mama Rocks Burgers",
            "Dormans Coffee", "Spring Valley Oven", "Wasp and Sprout", "Boho Eatery"
        ],
        "Luxury": [
            "Oaks and Corks Wines and Spirits", "Chupachap Wines", "The Alchemist Bar",
            "Brew Bistro and Lounge", "Quiver Lounge", "Club 1824 Bar and Restaurant",
            "Milan Lounge", "Barkeep Wines and Spirits", "Wines and Spirits Wholesale",
            "The Wine Shop", "K1 Klubhouse Sports Bar", "The Bar Next Door", "Havana Bar",
            "Black Diamond Bar", "Whiskey River Lounge", "B-Club Lounge", "Sky Lounge",
            "Cloud 9 Rooftop Bar", "Vintage Wine Cellar", "Corner Wines and Spirits",
            "Sunset Lounge", "Galaxy Bar and Grill", "Safari Lounge", "Tipsy Bar",
            "Grapevine Wines", "City Wine Merchants", "Oasis Lounge", "High Spirits Wines",
            "Level 7 Lounge", "Vineyard Wines", "Cellar Bar", "Boutique Wine Lounge",
            "Executive Lounge", "Terrace Bar and Lounge", "Mojito Bar", "Rumors Lounge",
            "Altitude Rooftop Lounge", "Gravity Bar", "1824 Bar", "Klub Lounge", "Beer and Wine"
        ],
        "Groceries": [
            "Naivas Supermarket", "Quickmart Supermarket", "Carrefour Hypermarket",
            "Chandarana Foodplus", "Cleanshelf Supermarket", "Eastmatt Supermarket",
            "Tuskys Supermarket", "Tumaini Supermarket", "Society Stores", "Magunas Supermarket",
            "Mama Mboga Vegetables Kiosk", "Green Grocers Fresh Produce", "City Butchery",
            "Prime Cuts Butchery", "Fresh Farm Produce", "Wholesale Provisions Kiosk",
            "Organic Veggies Mart", "Poultry and Meat Supplies", "Fruit Parlour",
            "Farmers Choice Pork Centre", "Neema Butchery", "Uchumi Supermarket",
            "Kasatt Supermarket", "Chieni Supermarket", "Mathai Supermarket", "Food Plus Mart"
        ],
        "Utilities": [
            "KPLC Prepaid Meter Token", "Kenya Power and Lighting Token", "KPLC Postpaid Bill",
            "Nairobi City Water and Sewerage NCWSC", "Zuku Fiber Home Internet",
            "Safaricom Home Fibre", "Faiba 4G Internet Jamii", "Mawingu WiFi Networks",
            "Poa Internet Kenya", "Eldoret Water ELDOWAS", "Kisumu Water KIWASCO",
            "Nakuru Water NAWASSCO", "Mombasa Water MOWASSCO", "Nyeri Water NYEWASCO",
            "Embu Water EWASCO", "Machakos Water MEWASS", "Coast Water Services"
        ],
        "Transport": [
            "TotalEnergies Service Station", "Shell Fuel Station", "Rubis Energy Kenya",
            "Ola Energy Petrol Station", "Astrol Petroleum", "National Oil NOCK",
            "Hass Petroleum", "Gulf Energy", "Be Energy Petrol", "Lexo Energy",
            "Uber BV Trip help.uber.com", "Bolt Kenya Ride", "Little Cab Kenya",
            "Super Metro Sacco Bus Fare", "2NK Sacco Matatu Fare", "4NTE Sacco Shuttle",
            "Mololine Sacco Shuttle", "Transline Classic Coach", "Modern Coast Express",
            "Easy Coach Bus", "Tahmeed Coach Express", "North Rift Shuttle",
            "Kinatwa Sacco", "KBS Bus Fare", "City Hoppa Bus Fare", "Metro Trans Shuttle"
        ],
        "Healthcare": [
            "Aga Khan University Hospital", "The Nairobi Hospital Outpatient",
            "Gertrude's Children's Hospital", "Mater Misericordiae Hospital",
            "MP Shah Hospital", "The Karen Hospital", "Kenyatta National Hospital KNH",
            "Avenue Hospital", "Coptic Mission Hospital", "Goodlife Pharmacy",
            "Haltons Pharmacy", "Medplus Pharmacy", "Portal Pharmacy", "City Chemist",
            "Care Chemists", "Metro Chemist", "PathCare Kenya Labs", "Lancet Laboratories",
            "Bliss Medical Centre", "Clinix Healthcare", "Nairobi West Hospital"
        ],
        "Telecommunications": [
            "Safaricom Airtime Topup direct", "Safaricom Postpaid Monthly Bill",
            "Safaricom Data Bundles", "Airtel Money Airtime topup",
            "Telkom Kenya Airtime topup", "Faiba Mobile Airtime bundles",
            "Safaricom Tunukiwa Bundles", "Airtel Tubonge Monthly Pack"
        ],
        "Personal Transfers": [],
        "Gambling": [
            "Sportpesa Betting Wallet", "Betika Kenya Betting", "Odibets Betting Deposit",
            "Mozzartbet Kenya Paybill", "1xBet Casino and Sportsbook", "Shabiki Bet Online",
            "Betway Sportsbook Kenya", "22Bet Kenya Online", "Bangbet Sports Betting",
            "Kwikbet Kenya Deposit", "Melbet Online Betting", "Lucky Box Casino",
            "Millionaires Casino Nairobi", "Mayfair Casino Westlands", "Finix Casino Hurlingham",
            "Captain's Club Casino", "Golden Key Casino", "Betpawa Kenya Deposit"
        ],
        "Loan Repayment": [
            "Hustler Fund Loan Repayment", "Tala Kenya Mobile Loan", "Branch International Loan",
            "Zenka Finance Mobile Loan", "Timiza Loan Repayment NCBA", "M-Shwari Loan Repayment",
            "KCB MPESA Loan Repayment", "Okash Mobile Loan App", "Platinum Credit Emergency Loan",
            "IPesa Mobile Loan", "Zash Loan Repayment", "Berry Mobile Loan", "Mogo Auto Loan",
            "Watu Credit Motorcycle Loan", "Asa International Microfinance", "Faulu Microfinance Loan",
            "KWFT Microfinance Loan", "Sidian Bank Loan Repayment", "Izwe Loans Kenya",
            "Equity Bank Eazzy Loan", "Co-op Bank MCo-op Cash Loan", "Absa Timiza Loan"
        ],
        "Savings & Investments": [
            "CIC Money Market Fund MMF", "Sanlam Unit Trust Investment", "Britam Money Market Fund",
            "ICEA LION Money Market Fund", "Old Mutual MMF Investment", "Dry Associates MMF",
            "Zimele Unit Trust Savings", "Co-op Money Market Fund", "Chumz Savings App Deposit",
            "Stima Sacco Monthly Shares", "Harambee Sacco Welfare Deposit", "Kenya Police Sacco",
            "Safaricom Sacco Shares", "Mhasibu Sacco Investment", "Mali Safaricom Wealth Management",
            "Standard Chartered Wealth MMF", "NCBA Unit Trust Fund", "Genghis Capital MMF",
            "Etica Money Market Fund", "Apollo Money Market Fund", "Ndovu Wealth Investments"
        ],
        "Rent & Housing": [
            "Lloyd Masika Property Rent", "Hass Consult Rent and Service Charge",
            "Villa Care Property Management", "Knight Frank Kenya Residential",
            "Regent Real Estate Rent", "Ridgeways Estate Rent Payment",
            "Kilimani Heights Apartments", "Westlands Crest Apartments",
            "Karen Green Estate Rent", "Lavington Court Apartments", "South B Apartments Rent",
            "Roysambu Heights Housing", "Ruiru Housing Cooperative", "Crystal Rivers Apartments",
            "Greatwall Gardens Housing Rent", "Parkview Apartments Kileleshwa",
            "Pinnacle Properties Rent", "Broll Property Rent", "Dunhill Consulting Rent", "PALM",
        ],
        "Education": [
            "University of Nairobi School Fees", "Strathmore University Tuition",
            "Kenyatta University Fees", "JKUAT Student Account", "Daystar University Tuition",
            "USIU Africa University Fees", "Mount Kenya University MKU", "Catholic University CUEA",
            "Brookhouse School Fees", "Makini School Tuition", "Braeburn School Fees",
            "St Marys School Nairobi", "Riara Group of Schools Fees", "Nairobi Academy School",
            "Alliance High School Fees", "Mang'u High School Fees", "Moi Girls High School",
            "Lenana School Fees", "Kenya High School Fees", "Rusinga School Fees",
            "Nova Pioneer Schools", "Peponi School Fees", "Hillcrest International School"
        ],
        "Government & Taxes": [
            "eCitizen Services Paybill 222222", "KRA iTax Tax Payment Paybill 572572",
            "KRA Customs and Excise Duty", "NTSA Driving Licence Inspection 222222",
            "National Hospital Insurance Fund NHIF 200222", "Social Health Authority SHA Healthcare",
            "National Social Security Fund NSSF 333300", "Higher Education Loans Board HELB 200800",
            "Nairobi City County Land Rates Paybill 224974", "Mombasa County Parking and Rates",
            "Kiambu County Revenue Services", "Nakuru County Revenue Services",
            "Machakos County Revenue Collection", "Judiciary e-Filing Filing Fees"
        ],
        "Business & Trade": [
            "Crown Paints Kenya Wholesale", "Bamburi Cement Supplies", "Kaluworks Aluminum Supplies",
            "Unga Farm Care Animal Feeds", "Kapa Oil Refineries Wholesale", "Bidco Africa Supplies",
            "Chui Hardware Building Supplies", "Twiga Foods Wholesale Fresh Produce",
            "Apex Steel Wholesale Distributors", "Devki Steel Mills Building Supplies",
            "General Hardware Construction Materials", "Kenpoly Plastics Supplies",
            "Nairobi Packaging Solutions", "Brookside Dairy Bulk Purchase"
        ]
    }

    return locations, first_names, last_names, seeds


def generate_mpesa_records(count_target: int = 50000, seed: int = 42) -> list:
    """Generates 50,000 M-Pesa specific narrative records across all categories."""
    random.seed(seed)
    locations, first_names, last_names, seeds = get_vocabularies()
    categories = list(seeds.keys())
    per_category = count_target // len(categories) + 1

    records = []
    for cat in categories:
        brands = seeds[cat]
        for _ in range(per_category):
            loc = random.choice(locations)
            till = random.randint(10000, 999999)
            acc = random.randint(100000, 9999999)

            if cat == "Personal Transfers":
                fn = random.choice(first_names)
                ln = random.choice(last_names)
                phone_prefix = random.choice(["07", "01", "+2547", "+2541", "2547"])
                phone = f"{phone_prefix}{random.randint(10000000, 99999999)}"
                token = f"[PHONE_ID_{random.randint(1000000000, 9999999999):x}]"
                patterns = [
                    f"{fn} {ln} {phone}",
                    f"Sent to {fn} {ln} {token}",
                    f"Received from {fn} {ln} {phone}",
                    f"{fn.upper()} {ln.upper()} {phone}",
                    f"Transfer to {fn} {ln} {token}",
                ]
                text = random.choice(patterns)
            elif cat == "Luxury":
                brand = random.choice(brands)
                patterns = [
                    f"Paid to {brand} {loc}",
                    f"{brand} Till {till}",
                    f"Paybill {till} {brand} {loc}",
                    f"{loc} {brand} Wines and Spirits",
                    f"{brand} Bar & Lounge Till {till}",
                ]
                text = random.choice(patterns)
            else:
                brand = random.choice(brands)
                patterns = [
                    f"Paid to {brand} {loc}",
                    f"{brand} Till {till}",
                    f"{brand} for account {acc}",
                    f"Paybill {till} {brand} {loc}",
                    f"Payment to {brand} {loc} Branch",
                ]
                text = random.choice(patterns)

            # Random casing
            r = random.random()
            if r < 0.35:
                text = text.upper()
            elif r < 0.60:
                text = text.lower()

            records.append((text, cat))

    random.shuffle(records)
    return records[:count_target]


def generate_bank_records(count_target: int = 50000, seed: int = 101) -> list:
    """Generates 50,000 Bank Statement narrative records across all categories."""
    random.seed(seed)
    locations, first_names, last_names, seeds = get_vocabularies()
    categories = list(seeds.keys())
    per_category = count_target // len(categories) + 1

    records = []
    for cat in categories:
        brands = seeds[cat]
        for _ in range(per_category):
            loc = random.choice(locations)
            acc_num = random.randint(1000000000, 9999999999)
            ref_num = f"FT{random.randint(100000, 999999)}"

            if cat == "Personal Transfers":
                fn = random.choice(first_names)
                ln = random.choice(last_names)
                acc_mask = f"[ACC_ID_{random.randint(1000000, 9999999):x}]"
                patterns = [
                    f"PESALINK TO {acc_mask} {fn.upper()} {ln.upper()}",
                    f"EFT TRANSFER {fn} {ln} {acc_mask}",
                    f"RTGS INWARD REMITTANCE {fn.upper()} {ln.upper()}",
                    f"SALARY CREDIT FROM ACME TECH TO {fn} {ln}",
                    f"FUNDS TRANSFER {fn} {ln} REF {ref_num}",
                    f"INTERBANK TRANSFER {fn.upper()} {ln.upper()}",
                ]
                text = random.choice(patterns)
            elif cat == "Luxury":
                brand = random.choice(brands)
                patterns = [
                    f"POS PURCHASE {brand.upper()} {loc.upper()}",
                    f"VISA DEBIT {brand} {loc}",
                    f"CARD TXN {brand.upper()} {ref_num}",
                    f"POINT OF SALE {brand} BAR & LOUNGE",
                    f"DIRECT DEBIT {brand} WINES AND SPIRITS",
                    f"POS {brand.upper()}",
                ]
                text = random.choice(patterns)
            else:
                brand = random.choice(brands)
                patterns = [
                    f"POS PURCHASE {brand.upper()} {loc.upper()}",
                    f"VISA DEBIT {brand} {loc}",
                    f"EFT PAYMENT {brand.upper()} {acc_num}",
                    f"STANDING ORDER {brand} REF {ref_num}",
                    f"DIRECT DEBIT {brand.upper()} ACC {acc_num}",
                    f"MBANK C2B {brand} {loc}",
                    f"BILL PAYMENT {brand.upper()} {ref_num}",
                    f"PESALINK TO {brand.upper()}",
                ]
                text = random.choice(patterns)

            # Random casing
            r = random.random()
            if r < 0.50:
                text = text.upper()
            elif r < 0.75:
                text = text.title()
            else:
                text = text.lower()

            records.append((text, cat))

    random.shuffle(records)
    return records[:count_target]


def generate_100k_dataset(seed: int = 42) -> pd.DataFrame:
    """Combines 50,000 M-Pesa + 50,000 Bank Statement records = 100,000 records."""
    print("Generating 50,000 M-Pesa transaction records...")
    mpesa_records = generate_mpesa_records(count_target=50000, seed=seed)
    print("Generating 50,000 Bank Statement transaction records...")
    bank_records = generate_bank_records(count_target=50000, seed=seed + 1)

    all_records = mpesa_records + bank_records
    random.seed(seed)
    random.shuffle(all_records)

    df = pd.DataFrame(all_records, columns=["counterparty_text", "category"])
    return df


def train_and_export():
    start_time = time.time()
    print("=" * 65)
    print("  TRAINING M-PESA & BANK STATEMENT UNIFIED CLASSIFICATION ENGINE")
    print("=" * 65)

    df = generate_100k_dataset(seed=42)
    print(f"\nTotal Dataset Size: {len(df):,} records ({df['category'].nunique()} categories)")

    print("\nCategory Distribution:")
    for cat, cnt in df["category"].value_counts().items():
        print(f"  • {cat:22s}: {cnt:6,d} records")

    X = df["counterparty_text"]
    y = df["category"]

    # 80/20 train/test split: 80,000 train / 20,000 validation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    print(f"\nTrain Set: {len(X_train):,} records | Validation Set: {len(X_test):,} records")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=25000
        )),
        ("clf", LGBMClassifier(
            n_estimators=150,
            learning_rate=0.08,
            min_child_samples=20,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        ))
    ])

    print("\nTraining LightGBM Text Categorization Pipeline (parallel n_jobs=-1)...")
    train_start = time.time()
    pipeline.fit(X_train, y_train)
    train_duration = time.time() - train_start
    print(f"Model training completed in {train_duration:.2f} seconds.")

    y_pred = pipeline.predict(X_test)
    print("\n" + "=" * 65)
    print("=== TEST SET EVALUATION REPORT (20,000 UNSEEN SAMPLES) ===")
    print("=" * 65)
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    model_path = "mpesa_classifier.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    total_duration = time.time() - start_time
    print(f"Saved optimized compressed model artifact to '{model_path}' ({total_duration:.2f}s total run).")


if __name__ == "__main__":
    train_and_export()
