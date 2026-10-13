import itertools
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


def generate_large_synthetic_data(samples_per_category: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Synthesizes an extensive, realistic labeled dataset (19,500+ records)
    representing Kenyan financial counterparties, till numbers, paybills, and P2P formats.
    """
    random.seed(seed)
    np.random.seed(seed)

    # 1. Geographic Locations & Neighborhoods across Kenya
    locations = [
        "Westlands", "Kilimani", "Lavington", "Karen", "CBD Nairobi", "Parklands",
        "Roysambu", "Langata", "Ngong Road", "Thika Road", "Mombasa Road", "Buruburu",
        "South C", "South B", "Kasarani", "Ruiru", "Kiambu Town", "Nakuru Town",
        "Mombasa Nyali", "Mombasa Diani", "Kisumu Milimani", "Eldoret Town",
        "Kitengela", "Ongata Rongai", "Utawala", "Embakasi", "Donholm", "Githurai",
        "Kikuyu", "Naivasha", "Nanyuki", "Machakos", "Meru", "Kakamega", "Kericho",
        "Eastleigh", "Syokimau", "Kahawa Sukari", "Kahawa West", "Upper Hill"
    ]

    # 2. Common Kenyan First and Last Names
    first_names = [
        "John", "Mary", "Peter", "Jane", "James", "Grace", "David", "Faith", "Joseph",
        "Mercy", "Daniel", "Esther", "Samuel", "Joy", "Paul", "Brian", "Kevin", "Denis",
        "Evans", "Victor", "Lucy", "Beatrice", "Rose", "Christine", "Sarah", "Mohammed",
        "Fatuma", "Hassan", "Amina", "Ali", "Ahmed", "Hussein", "Abdullahi", "Omar",
        "Collins", "Geoffrey", "Caroline", "Jackline", "Sharon", "Kennedy", "Kelvin",
        "Emmanuel", "Francis", "Stephen", "Martin", "Patrick", "Purity", "Brenda"
    ]
    last_names = [
        "Kamau", "Wanjiku", "Njoroge", "Wangari", "Kariuki", "Mwangi", "Maina", "Kimani",
        "Nyambura", "Ochieng", "Odhiambo", "Otieno", "Anyango", "Atieno", "Omondi",
        "Kiprop", "Cheruiyot", "Kiprono", "Chebet", "Rotich", "Korir", "Chemutai",
        "Mutua", "Musyoka", "Mutiso", "Mwende", "Kilonzo", "Wamalwa", "Simiyu", "Wafula",
        "Nekesa", "Barasa", "Juma", "Karanja", "Muthoni", "Gacheru", "Ndung'u", "Ouma",
        "Wekesa", "Moraa", "Mogaka", "Makori", "Onyango", "Achieng", "Koech"
    ]

    # 3. Category Seed Vocabularies
    seeds = {
        # Food & Dining (Diverse food entities, restaurants, cafes, fast food, bakeries, nyama choma)
        "Food & Dining": {
            "brands": [
                "Java House", "Artcaffe", "KFC", "Dominos Pizza", "CJ Restaurant", "CJ's",
                "Big Square", "Galitos", "Pizza Inn", "Chicken Inn", "Creamy Inn", "Subway",
                "Steers", "Debonairs Pizza", "Planet Yogurt", "Mama Otieno Fish Point",
                "Kilimanjaro Restaurant", "Swahili Dishes", "City Nyama Choma", "Charcoal Grill",
                "Broadway Bakery Delight", "Campus Canteen", "Tacos and Burgers", "Coffee Point",
                "Urban Eatery", "Kempinski Cafe", "Sarova Restaurant", "Intercontinental Diner",
                "Nairobi Kitchen", "Pizza Mojo", "Kuku Foods", "Nyama Bite", "Roast House",
                "Sweet Treats Bakery", "Bake House", "Choma Zone", "Fish Grill", "Shawarma Hub",
                "Mama Nilishe Eatery", "Wimpy Fast Food", "Toranj Persian Restaurant", "Habesha",
                "Fogo Gaucho Brazilian", "Talisman Restaurant", "About Thyme", "Mama Rocks Burgers"
            ],
            "suffixes": [
                "Restaurant", "Cafe", "Café", "Eatery", "Diner", "Fast Food", "Kitchen",
                "Bakery", "Grill", "Nyama Choma", "Fish Point", "Coffee Shop", "Canteen",
                "Food Court", "Bites", "Catering", "Pizzeria", "Dishes", "Meals"
            ]
        },

        # Luxury (Strictly contains 'wine', 'wines', 'bar', or 'lounge')
        "Luxury": {
            "brands": [
                "Oaks and Corks Wines and Spirits", "Chupachap Wines", "The Alchemist Bar",
                "Brew Bistro and Lounge", "Quiver Lounge", "Club 1824 Bar and Restaurant",
                "Milan Lounge", "Barkeep Wines and Spirits", "Wines and Spirits Wholesale",
                "The Wine Shop", "K1 Klubhouse Sports Bar", "The Bar Next Door", "Havana Bar",
                "Black Diamond Bar", "Whiskey River Lounge", "B-Club Lounge", "Sky Lounge",
                "Cloud 9 Rooftop Bar", "Vintage Wine Cellar", "Corner Wines and Spirits",
                "Sunset Lounge", "Galaxy Bar and Grill", "Safari Lounge", "Tipsy Bar",
                "Grapevine Wines", "City Wine Merchants", "Oasis Lounge", "High Spirits Wines",
                "Level 7 Lounge", "The Alchemist Bar", "Vineyard Wines", "Cellar Bar",
                "Boutique Wine Lounge", "Executive Lounge", "Terrace Bar and Lounge",
                "Mojito Bar", "Rumors Lounge", "Altitude Rooftop Lounge", "Gravity Bar"
            ],
            "suffixes": [
                "Wines and Spirits", "Wines & Spirits", "Wine Shop", "Wine Cellar", "Bar",
                "Sports Bar", "Cocktail Bar", "Lounge", "Rooftop Lounge", "VIP Lounge",
                "Bar and Restaurant", "Lounge and Grill", "Wine Merchants"
            ]
        },

        # Groceries (Supermarkets, greengrocers, fresh markets, butcheries)
        "Groceries": {
            "brands": [
                "Naivas Supermarket", "Quickmart Supermarket", "Carrefour Hypermarket",
                "Chandarana Foodplus", "Cleanshelf Supermarket", "Eastmatt Supermarket",
                "Tuskys Supermarket", "Tumaini Supermarket", "Society Stores", "Magunas Supermarket",
                "Mama Mboga Vegetables Kiosk", "Green Grocers Fresh Produce", "City Butchery",
                "Prime Cuts Butchery", "Fresh Farm Produce", "Wholesale Provisions Kiosk",
                "Organic Veggies Mart", "Poultry and Meat Supplies", "Fruit Parlour",
                "Farmers Choice Pork Centre", "Neema Butchery", "Uchumi Supermarket",
                "Kasatt Supermarket", "Chieni Supermarket", "Mathai Supermarket"
            ],
            "suffixes": [
                "Supermarket", "Supermarket Till", "Mini Market", "Grocery", "Groceries",
                "Vegetables Kiosk", "Greengrocers", "Butchery", "Fresh Meat", "Provisions Store"
            ]
        },

        # Utilities (Essential fixed overhead)
        "Utilities": {
            "brands": [
                "KPLC Prepaid Meter Token", "Kenya Power and Lighting Token", "KPLC Postpaid Bill",
                "Nairobi City Water and Sewerage NCWSC", "Zuku Fiber Home Internet",
                "Safaricom Home Fibre", "Faiba 4G Internet Jamii", "Mawingu WiFi Networks",
                "Poa Internet Kenya", "Eldoret Water ELDOWAS", "Kisumu Water KIWASCO",
                "Nakuru Water NAWASSCO", "Mombasa Water MOWASSCO", "Nyeri Water NYEWASCO",
                "Embu Water EWASCO", "Machakos Water MEWASS", "Coast Water Services"
            ],
            "suffixes": [
                "Prepaid Token", "Bill Payment", "Water Bill", "Internet Subscription",
                "Monthly Bundle", "Utility Paybill", "Meter Token 482910"
            ]
        },

        # Transport & Fuel (Fuel stations, Ride-hailing, Matatus, Coaches)
        "Transport": {
            "brands": [
                "TotalEnergies Service Station", "Shell Fuel Station", "Rubis Energy Kenya",
                "Ola Energy Petrol Station", "Astrol Petroleum", "National Oil NOCK",
                "Hass Petroleum", "Gulf Energy", "Be Energy Petrol", "Lexo Energy",
                "Uber BV Trip help.uber.com", "Bolt Kenya Ride", "Little Cab Kenya",
                "Super Metro Sacco Bus Fare", "2NK Sacco Matatu Fare", "4NTE Sacco Shuttle",
                "Mololine Sacco Shuttle", "Transline Classic Coach", "Modern Coast Express",
                "Easy Coach Bus", "Tahmeed Coach Express", "North Rift Shuttle",
                "Kinatwa Sacco", "KBS Bus Fare", "City Hoppa Bus Fare", "Metro Trans Shuttle"
            ],
            "suffixes": [
                "Service Station", "Petrol Pump", "Fuel Station", "Bus Fare", "Matatu Fare",
                "Express Shuttle", "Travel Ticket", "Ride Till", "Fleet Service"
            ]
        },

        # Healthcare (Hospitals, Pharmacies, Chemists, Diagnostics)
        "Healthcare": {
            "brands": [
                "Aga Khan University Hospital", "The Nairobi Hospital Outpatient",
                "Gertrude's Children's Hospital", "Mater Misericordiae Hospital",
                "MP Shah Hospital", "The Karen Hospital", "Kenyatta National Hospital KNH",
                "Avenue Hospital", "Coptic Mission Hospital", "Goodlife Pharmacy",
                "Haltons Pharmacy", "Medplus Pharmacy", "Portal Pharmacy", "City Chemist",
                "Care Chemists", "Metro Chemist", "PathCare Kenya Labs", "Lancet Laboratories",
                "Bliss Medical Centre", "Clinix Healthcare", "Nairobi West Hospital"
            ],
            "suffixes": [
                "Pharmacy", "Chemist", "Clinic", "Hospital", "Outpatient Care", "Dispensary",
                "Laboratory", "Diagnostics", "Medical Centre", "Dental Clinic"
            ]
        },

        # Telecommunications (Airtime, Mobile Bundles)
        "Telecommunications": {
            "brands": [
                "Safaricom Airtime Topup direct", "Safaricom Postpaid Monthly Bill",
                "Safaricom Data Bundles", "Airtel Money Airtime topup",
                "Telkom Kenya Airtime topup", "Faiba Mobile Airtime bundles",
                "Safaricom Tunukiwa Bundles", "Airtel Tubonge Monthly Pack"
            ],
            "suffixes": [
                "Airtime Topup", "Airtime Purchase", "Data Bundles", "Postpaid Bill",
                "Monthly Pack", "Airtime Direct", "Mobile Topup"
            ]
        },

        # Personal Transfers (P2P Social & Remittance)
        "Personal Transfers": {
            "brands": [],  # Generated dynamically using Kenyan first/last names & phone variations
            "suffixes": []
        },

        # Gambling & Betting (High Credit Risk Indicator)
        "Gambling": {
            "brands": [
                "Sportpesa Betting Wallet", "Betika Kenya Betting", "Odibets Betting Deposit",
                "Mozzartbet Kenya Paybill", "1xBet Casino and Sportsbook", "Shabiki Bet Online",
                "Betway Sportsbook Kenya", "22Bet Kenya Online", "Bangbet Sports Betting",
                "Kwikbet Kenya Deposit", "Melbet Online Betting", "Lucky Box Casino",
                "Millionaires Casino Nairobi", "Mayfair Casino Westlands", "Finix Casino Hurlingham",
                "Captain's Club Casino", "Golden Key Casino", "Betpawa Kenya Deposit"
            ],
            "suffixes": [
                "Betting Deposit", "Wallet Deposit", "Sportsbook Token", "Paybill Deposit",
                "Casino Account", "Lottery Token", "Online Stake"
            ]
        },

        # Loan Repayment (Mobile loans, SACCO loans, microfinance)
        "Loan Repayment": {
            "brands": [
                "Hustler Fund Loan Repayment", "Tala Kenya Mobile Loan", "Branch International Loan",
                "Zenka Finance Mobile Loan", "Timiza Loan Repayment NCBA", "M-Shwari Loan Repayment",
                "KCB MPESA Loan Repayment", "Okash Mobile Loan App", "Platinum Credit Emergency Loan",
                "IPesa Mobile Loan", "Zash Loan Repayment", "Berry Mobile Loan", "Mogo Auto Loan",
                "Watu Credit Motorcycle Loan", "Asa International Microfinance", "Faulu Microfinance Loan",
                "KWFT Microfinance Loan", "Sidian Bank Loan Repayment", "Izwe Loans Kenya"
            ],
            "suffixes": [
                "Loan Repayment", "Mobile Loan Payoff", "Loan Payment", "Emergency Loan",
                "Loan Settlement", "Installment Payment", "Microfinance Loan"
            ]
        },

        # Savings & Investments (Saccos, MMFs, Wealth Management)
        "Savings & Investments": {
            "brands": [
                "CIC Money Market Fund MMF", "Sanlam Unit Trust Investment", "Britam Money Market Fund",
                "ICEA LION Money Market Fund", "Old Mutual MMF Investment", "Dry Associates MMF",
                "Zimele Unit Trust Savings", "Co-op Money Market Fund", "Chumz Savings App Deposit",
                "Stima Sacco Monthly Shares", "Harambee Sacco Welfare Deposit", "Kenya Police Sacco",
                "Safaricom Sacco Shares", "Mhasibu Sacco Investment", "Mali Safaricom Wealth Management",
                "Standard Chartered Wealth MMF", "NCBA Unit Trust Fund", "Genghis Capital MMF",
                "Etica Money Market Fund", "Apollo Money Market Fund", "Ndovu Wealth Investments"
            ],
            "suffixes": [
                "Money Market Fund", "MMF Deposit", "Unit Trust Investment", "Monthly Shares",
                "Sacco Deposit", "Wealth Management", "Investment Deposit", "Savings Account"
            ]
        },

        # Rent & Housing (Real estate, apartments, property management)
        "Rent & Housing": {
            "brands": [
                "Lloyd Masika Property Rent", "Hass Consult Rent and Service Charge",
                "Villa Care Property Management", "Knight Frank Kenya Residential",
                "Regent Real Estate Rent", "Ridgeways Estate Rent Payment",
                "Kilimani Heights Apartments", "Westlands Crest Apartments",
                "Karen Green Estate Rent", "Lavington Court Apartments", "South B Apartments Rent",
                "Roysambu Heights Housing", "Ruiru Housing Cooperative", "Crystal Rivers Apartments",
                "Greatwall Gardens Housing Rent", "Parkview Apartments Kileleshwa",
                "Pinnacle Properties Rent", "Broll Property Rent", "Dunhill Consulting Rent"
            ],
            "suffixes": [
                "Rent Payment", "Monthly Rent", "Service Charge and Rent", "Apartments Rent",
                "Property Management", "Estate Rent", "House Rent", "Residential Lease"
            ]
        },

        # Education (Schools, Academies, Colleges, Universities)
        "Education": {
            "brands": [
                "University of Nairobi School Fees", "Strathmore University Tuition",
                "Kenyatta University Fees", "JKUAT Student Account", "Daystar University Tuition",
                "USIU Africa University Fees", "Mount Kenya University MKU", "Catholic University CUEA",
                "Brookhouse School Fees", "Makini School Tuition", "Braeburn School Fees",
                "St Marys School Nairobi", "Riara Group of Schools Fees", "Nairobi Academy School",
                "Alliance High School Fees", "Mang'u High School Fees", "Moi Girls High School",
                "Lenana School Fees", "Kenya High School Fees", "Rusinga School Fees",
                "Nova Pioneer Schools", "Peponi School Fees", "Hillcrest International School"
            ],
            "suffixes": [
                "School Fees", "Tuition Fee", "Tuition and Exam", "Term 1 Fees", "Term 2 Fees",
                "Term 3 Fees", "Student Account", "Admission Fees", "Academic Tuition"
            ]
        }
    }

    records = []

    for category, content in seeds.items():
        count = 0
        while count < samples_per_category:
            if category == "Personal Transfers":
                # Generate realistic P2P records
                fn = random.choice(first_names)
                ln = random.choice(last_names)
                # Formats: Phone with 07xx/01xx/+254xx or masked [PHONE_ID_...]
                phone_prefix = random.choice(["07", "01", "+2547", "+2541", "2547"])
                phone_body = f"{random.randint(10000000, 99999999)}"
                phone_str = f"{phone_prefix}{phone_body}"
                masked_token = f"[PHONE_ID_{random.randint(1000000000, 9999999999):x}]"
                
                template_choice = random.randint(1, 5)
                if template_choice == 1:
                    text = f"{fn} {ln} {phone_str}"
                elif template_choice == 2:
                    text = f"{fn} {ln} {masked_token}"
                elif template_choice == 3:
                    text = f"{fn} {ln}"
                elif template_choice == 4:
                    text = f"{fn.upper()} {ln.upper()} {phone_str}"
                else:
                    text = f"Payment to {fn} {ln} {masked_token}"

            elif category == "Luxury":
                # Guarantee presence of 'wine', 'wines', 'bar', or 'lounge'
                brand = random.choice(content["brands"])
                loc = random.choice(locations)
                till_num = random.randint(10000, 999999)
                suffix = random.choice(content["suffixes"])
                
                style = random.randint(1, 4)
                if style == 1:
                    text = f"{brand} {loc}"
                elif style == 2:
                    text = f"{brand} Till {till_num}"
                elif style == 3:
                    text = f"{loc} {suffix} Till {till_num}"
                else:
                    text = f"Payment to {brand} {loc}"

            else:
                brand = random.choice(content["brands"])
                loc = random.choice(locations)
                till_num = random.randint(10000, 999999)
                acc_num = random.randint(100000, 9999999)
                suffix = random.choice(content["suffixes"])
                
                style = random.randint(1, 5)
                if style == 1:
                    text = f"{brand} {loc}"
                elif style == 2:
                    text = f"{brand} Till {till_num}"
                elif style == 3:
                    text = f"{brand} acc {acc_num}"
                elif style == 4:
                    text = f"{loc} {suffix} Till {till_num}"
                else:
                    text = f"Payment to {brand} {loc} Branch"

            # Apply realistic text casing and formatting noise
            casing_style = random.randint(1, 4)
            if casing_style == 1:
                clean_text = text.upper()
            elif casing_style == 2:
                clean_text = text.lower()
            elif casing_style == 3:
                clean_text = text.title()
            else:
                clean_text = text

            records.append((clean_text, category))
            count += 1

    df = pd.DataFrame(records, columns=["counterparty_text", "category"])
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def train_and_export():
    start_time = time.time()
    print("=" * 60)
    print("Generating large-scale synthetic Kenyan financial dataset...")
    df = generate_large_synthetic_data(samples_per_category=1600, seed=42)
    print(f"Dataset generated: {len(df):,} total samples across {df['category'].nunique()} categories.")
    
    print("\nSamples per category distribution:")
    for cat, cnt in df["category"].value_counts().items():
        print(f"  - {cat:22s}: {cnt:5,d} samples")

    X = df["counterparty_text"]
    y = df["category"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nSplit: {len(X_train):,} training records, {len(X_test):,} validation records.")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            min_df=2,
            sublinear_tf=True,
            max_features=15000
        )),
        ("clf", LGBMClassifier(
            n_estimators=150,
            learning_rate=0.08,
            min_child_samples=10,
            random_state=42,
            verbose=-1,
            n_jobs=-1
        ))
    ])

    print("\nTraining optimized TF-IDF + LightGBM pipeline...")
    train_start = time.time()
    pipeline.fit(X_train, y_train)
    train_duration = time.time() - train_start
    print(f"Model training completed in {train_duration:.2f} seconds.")

    y_pred = pipeline.predict(X_test)
    print("\n" + "=" * 60)
    print("=== TEST SET EVALUATION REPORT (4,160 unseen samples) ===")
    print("=" * 60)
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    model_path = "mpesa_classifier.joblib"
    joblib.dump(pipeline, model_path, compress=3)
    total_duration = time.time() - start_time
    print(f"Saved optimized model artifact to '{model_path}' ({total_duration:.2f}s total run).")


if __name__ == "__main__":
    train_and_export()
