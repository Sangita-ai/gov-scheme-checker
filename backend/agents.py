"""
AI agent functions for the Government Scheme Eligibility Checker.
Uses Groq API (primary) with Gemini Flash (fallback).
No LangChain/LangGraph — plain Python with direct API calls.
"""

import os
import json
import re
from typing import Optional


def call_llm(prompt: str, system: str, max_tokens: int = 500) -> str:
    """
    Try Groq (llama-3.1-8b-instant) first.
    Fallback to Gemini Flash if Groq fails or rate-limits.
    Returns empty string on total failure.
    """
    groq_key = os.getenv("GROQ_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")

    if groq_key and groq_key != "your_groq_key_here":
        try:
            import groq as groq_sdk
            client = groq_sdk.Groq(api_key=groq_key)
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=max_tokens,
                temperature=0.3,
            )
            return response.choices[0].message.content
        except Exception as groq_error:
            print(f"⚠️  Groq failed: {groq_error}. Trying Gemini...")

    if gemini_key and gemini_key != "your_gemini_key_here":
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            full_prompt = f"{system}\n\n{prompt}"
            response = model.generate_content(full_prompt)
            return response.text
        except Exception as gemini_error:
            print(f"⚠️  Gemini also failed: {gemini_error}")

    return ""


def _safe_json_parse(text: str) -> Optional[dict]:
    """Extract and parse JSON from LLM output, handling markdown fences."""
    if not text:
        return None
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


PROFILE_FIELDS = [
    "age", "gender", "state", "occupation", "income_annual_inr",
    "caste_category", "is_bpl", "has_bank_account", "has_aadhar",
    "is_farmer", "land_owned_acres", "number_of_children", "has_disability",
]

FIELD_QUESTIONS = {
    "age":              "How old are you? (Please tell me your age in years)",
    "gender":           "Are you male, female, or other?",
    "state":            "Which state or union territory do you live in?",
    "occupation":       "What is your main occupation? (farmer / student / salaried / self-employed / unemployed / street-vendor / artisan / woman-homemaker)",
    "income_annual_inr":"What is your approximate annual family income in rupees? (It is okay to say 'don't know')",
    "caste_category":   "What is your caste category? (SC / ST / OBC / General / prefer-not-to-say)",
    "is_bpl":           "Does your family have a BPL (Below Poverty Line) ration card? (yes / no)",
    "has_bank_account": "Do you have a bank account? (yes / no)",
    "has_aadhar":       "Do you have an Aadhar card? (yes / no)",
    "is_farmer":        "Are you currently involved in farming or agricultural work? (yes / no)",
    "land_owned_acres": "How much agricultural land do you own or cultivate? (in acres — type 0 if none)",
    "number_of_children":"How many children do you have? (type a number, e.g. 0, 1, 2)",
    "has_disability":   "Do you have any disability? (yes / no)",
}

FIELD_QUESTIONS_HI = {
    "age":              "आपकी उम्र क्या है? (साल में बताएं)",
    "gender":           "आप पुरुष हैं, महिला हैं, या अन्य?",
    "state":            "आप किस राज्य में रहते हैं?",
    "occupation":       "आपका मुख्य काम क्या है? (किसान / छात्र / नौकरीपेशा / स्व-रोजगार / बेरोजगार / स्ट्रीट वेंडर / कारीगर / गृहिणी)",
    "income_annual_inr":"आपकी परिवार की सालाना आय लगभग कितनी है? (नहीं पता तो 'नहीं पता' कहें)",
    "caste_category":   "आपकी जाति श्रेणी क्या है? (SC / ST / OBC / General / बताना नहीं चाहते)",
    "is_bpl":           "क्या आपके परिवार के पास BPL राशन कार्ड है? (हाँ / नहीं)",
    "has_bank_account": "क्या आपके पास बैंक खाता है? (हाँ / नहीं)",
    "has_aadhar":       "क्या आपके पास आधार कार्ड है? (हाँ / नहीं)",
    "is_farmer":        "क्या आप खेती-बाड़ी का काम करते हैं? (हाँ / नहीं)",
    "land_owned_acres": "आपके पास कितनी कृषि भूमि है? (एकड़ में — अगर नहीं है तो 0 लिखें)",
    "number_of_children":"आपके कितने बच्चे हैं? (संख्या बताएं)",
    "has_disability":   "क्या आपको कोई विकलांगता है? (हाँ / नहीं)",
}

FIELD_QUESTIONS_BN = {
    "age":              "আপনার বয়স কত? (বছরে বলুন)",
    "gender":           "আপনি পুরুষ, মহিলা, না অন্য?",
    "state":            "আপনি কোন রাজ্যে থাকেন?",
    "occupation":       "আপনার প্রধান পেশা কী? (কৃষক / ছাত্র / চাকরিজীবী / স্ব-নিযুক্ত / বেকার / পথ বিক্রেতা / কারিগর / গৃহিণী)",
    "income_annual_inr":"আপনার পরিবারের বার্ষিক আয় কত? (না জানলে 'জানি না' বলুন)",
    "caste_category":   "আপনার জাতি বিভাগ কী? (SC / ST / OBC / General / বলতে চাই না)",
    "is_bpl":           "আপনার পরিবারে BPL রেশন কার্ড আছে? (হ্যাঁ / না)",
    "has_bank_account": "আপনার ব্যাংক অ্যাকাউন্ট আছে? (হ্যাঁ / না)",
    "has_aadhar":       "আপনার আধার কার্ড আছে? (হ্যাঁ / না)",
    "is_farmer":        "আপনি কি কৃষিকাজে যুক্ত? (হ্যাঁ / না)",
    "land_owned_acres": "আপনার কত কৃষি জমি আছে? (একরে — না থাকলে 0 লিখুন)",
    "number_of_children":"আপনার কতজন সন্তান আছে? (সংখ্যা বলুন)",
    "has_disability":   "আপনার কোনো প্রতিবন্ধিতা আছে? (হ্যাঁ / না)",
}

LANGUAGE_INSTRUCTIONS = {
    "hi": "तुम्हें हिंदी में जवाब देना है। सरल और मैत्रीपूर्ण भाषा का उपयोग करो।",
    "bn": "আপনাকে বাংলায় উত্তর দিতে হবে। সহজ এবং বন্ধুত্বপূর্ণ ভাষা ব্যবহার করুন।",
    "en": "Respond in simple, warm English.",
}

YES_PATTERNS = re.compile(
    r'(yes|yeah|yep|yup|sure|ok|okay|haan|ha\b|haa|han\b|ji\s*haan|\bji\b|bilkul|'
    r'হ্যাঁ|হ্যা|আছে|আছি|হ্যাঁ\s*আছে)',
    re.IGNORECASE | re.UNICODE
)
NO_PATTERNS = re.compile(
    r'\b(no|nope|nahi|nahin|naa|na|nhin|नहीं|नही|ना|না|নেই|নাই|না|করি\s*না|নেই\s*না)\b',
    re.IGNORECASE | re.UNICODE
)

STATE_KEYWORDS = {
    "west bengal": "West Bengal", "wb": "West Bengal", "bengal": "West Bengal",
    "paschim banga": "West Bengal", "পশ্চিমবঙ্গ": "West Bengal",
    "uttar pradesh": "Uttar Pradesh", "up": "Uttar Pradesh",
    "maharashtra": "Maharashtra", "mumbai": "Maharashtra",
    "delhi": "Delhi", "new delhi": "Delhi",
    "bihar": "Bihar", "rajasthan": "Rajasthan",
    "madhya pradesh": "Madhya Pradesh", "mp": "Madhya Pradesh",
    "tamil nadu": "Tamil Nadu", "karnataka": "Karnataka",
    "andhra pradesh": "Andhra Pradesh", "telangana": "Telangana",
    "gujarat": "Gujarat", "odisha": "Odisha", "orissa": "Odisha",
    "jharkhand": "Jharkhand", "chhattisgarh": "Chhattisgarh",
    "punjab": "Punjab", "haryana": "Haryana", "kerala": "Kerala",
    "assam": "Assam", "himachal pradesh": "Himachal Pradesh",
    "uttarakhand": "Uttarakhand", "goa": "Goa",
    "jammu and kashmir": "Jammu and Kashmir", "j&k": "Jammu and Kashmir",
    "manipur": "Manipur", "meghalaya": "Meghalaya", "mizoram": "Mizoram",
    "nagaland": "Nagaland", "sikkim": "Sikkim", "tripura": "Tripura",
    "arunachal pradesh": "Arunachal Pradesh",
}


OCCUPATION_KEYWORDS = {
    "farmer": "farmer", "farming": "farmer", "agriculture": "farmer",
    "kisan": "farmer", "किसान": "farmer", "কৃষক": "farmer",
    "student": "student", "studying": "student", "school": "student",
    "college": "student", "ছাত্র": "student", "छात्र": "student",
    "salaried": "salaried", "job": "salaried", "employee": "salaried",
    "government job": "salaried", "private job": "salaried",
    "self-employed": "self-employed", "business": "self-employed",
    "shop": "self-employed", "व्यापार": "self-employed",
    "unemployed": "unemployed", "no job": "unemployed", "jobless": "unemployed",
    "बेरोजगार": "unemployed", "বেকার": "unemployed",
    "street vendor": "street-vendor", "vendor": "street-vendor",
    "hawker": "street-vendor", "thela": "street-vendor",
    "artisan": "artisan", "craftsman": "artisan", "weaver": "artisan",
    "carpenter": "artisan", "blacksmith": "artisan", "potter": "artisan",
    "tailor": "artisan", "कारीगर": "artisan", "কারিগর": "artisan",
    "homemaker": "woman-homemaker", "housewife": "woman-homemaker",
    "ghar": "woman-homemaker", "घर": "woman-homemaker", "গৃহিণী": "woman-homemaker",
}


CASTE_KEYWORDS = {
    "sc": "SC", "scheduled caste": "SC", "dalit": "SC",
    "st": "ST", "scheduled tribe": "ST", "tribal": "ST", "adivasi": "ST",
    "obc": "OBC", "other backward": "OBC",
    "general": "General", "unreserved": "General", "open": "General",
    "forward": "General",
    "prefer not": "prefer-not-to-say", "don't want": "prefer-not-to-say",
    "not say": "prefer-not-to-say",
}


def _rule_based_extract(message: str, current_profile: dict, next_expected_field: str = None) -> dict:
    """
    Pure Python extraction BEFORE calling the LLM.
    Handles: numbers, yes/no booleans, states, occupations, caste.
    This is the KEY fix — LLM can no longer drop simple answers.
    """
    msg = message.strip()
    msg_lower = msg.lower()
    updates = {}

    # ── 1. Age ────────────────────────────────────────────────────
    if current_profile.get("age") is None:
        # Match patterns like "25", "25 years", "I am 25", "मेरी उम्र 25 है", "আমার বয়স ২৫"
        age_match = re.search(r'\b(\d{1,3})\s*(?:years?|साल|वर्ष|বছর|yr)?\b', msg)
        if age_match:
            age = int(age_match.group(1))
            if 5 <= age <= 100:
                updates["age"] = age

    
    if current_profile.get("gender") is None:
        if re.search(r'\b(male|man|boy|he|him|पुरुष|男|পুরুষ)\b', msg_lower):
            updates["gender"] = "male"
        elif re.search(r'\b(female|woman|girl|she|her|महिला|औरत|lady|महिला|মহিলা|নারী)\b', msg_lower):
            updates["gender"] = "female"

    if current_profile.get("state") is None:
        for keyword, state_name in STATE_KEYWORDS.items():
            if keyword in msg_lower:
                updates["state"] = state_name
                break

    if current_profile.get("occupation") is None:
        for keyword, occ in OCCUPATION_KEYWORDS.items():
            if keyword in msg_lower:
                updates["occupation"] = occ
                if occ == "farmer":
                    updates["is_farmer"] = True
                break

    
    if current_profile.get("income_annual_inr") is None:
        
        if re.search(r"don'?t\s*know|pata\s*nahi|nahi\s*pata|पता\s*नहीं|জানি\s*না|জানিনা", msg_lower):
            updates["income_annual_inr"] = 200000
        else:
            
            lakh_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:lakh|lac|लाख|লাখ)', msg_lower)
            k_match = re.search(r'(\d+(?:\.\d+)?)\s*(?:thousand|k\b|हज़ार|हजार|হাজার)', msg_lower)
            plain_match = re.search(r'\b(\d{4,9})\b', msg)  # 4-9 digit number = likely income

            if lakh_match:
                updates["income_annual_inr"] = float(lakh_match.group(1)) * 100000
            elif k_match:
                updates["income_annual_inr"] = float(k_match.group(1)) * 1000
            elif plain_match:
                val = int(plain_match.group(1))
                if 1000 <= val <= 99999999:
                    updates["income_annual_inr"] = float(val)

    
    if current_profile.get("caste_category") is None:
        for keyword, caste in CASTE_KEYWORDS.items():
            if keyword in msg_lower:
                updates["caste_category"] = caste
                break


    BOOL_FIELDS = ["is_bpl", "has_bank_account", "has_aadhar", "is_farmer", "has_disability"]

    
    target_bool_field = None
    if next_expected_field and next_expected_field in BOOL_FIELDS:
        target_bool_field = next_expected_field
    else:
        for bf in BOOL_FIELDS:
            if current_profile.get(bf) is None:
                if bf == "is_farmer" and current_profile.get("occupation") == "farmer":
                    continue
                target_bool_field = bf
                break

    if target_bool_field and current_profile.get(target_bool_field) is None:
        if YES_PATTERNS.search(msg_lower):
            updates[target_bool_field] = True
        elif NO_PATTERNS.search(msg_lower):
            updates[target_bool_field] = False

    
    if current_profile.get("land_owned_acres") is None:
        land_match = re.search(r'\b(\d+(?:\.\d+)?)\s*(?:acres?|एकड़|একর|bigha|बीघा|বিঘা)?\b', msg_lower)
        if land_match:
            val = float(land_match.group(1))
            if val <= 1000:  # sanity check
                updates["land_owned_acres"] = val
        elif re.search(r'\b(no\s*land|don\'?t\s*have|नहीं\s*है|জমি\s*নেই)\b', msg_lower):
            updates["land_owned_acres"] = 0.0

    
    if current_profile.get("number_of_children") is None:
        child_match = re.search(r'\b(\d{1,2})\s*(?:children|child|kids?|son|daughter|बच्चे|बच्चा|सन्तान|ছেলে|মেয়ে|সন্তান)?\b', msg_lower)
        if child_match:
            val = int(child_match.group(1))
            if val <= 20:
                updates["number_of_children"] = val
        elif re.search(r'\b(no\s*child|no\s*kid|childless|निःसंतान|কোনো\s*সন্তান\s*নেই)\b', msg_lower):
            updates["number_of_children"] = 0


    return updates


def _get_next_field(profile: dict) -> Optional[str]:
    """Return the name of the next field that still needs to be filled."""
    for field in PROFILE_FIELDS:
        if profile.get(field) is None:
            return field
    return None


def _get_question(field: str, language: str) -> str:
    """Return the question for a field in the correct language."""
    if language == "hi":
        return FIELD_QUESTIONS_HI.get(field, FIELD_QUESTIONS.get(field, ""))
    if language == "bn":
        return FIELD_QUESTIONS_BN.get(field, FIELD_QUESTIONS.get(field, ""))
    return FIELD_QUESTIONS.get(field, "")


def _confirmation_message(field: str, value, language: str) -> str:
    """Generate a brief acknowledgement when a field is saved."""
    if language == "hi":
        return f"ठीक है! "
    if language == "bn":
        return f"বুঝেছি! "
    return "Got it! "


def parse_input(
    user_message: str,
    current_profile: dict,
    chat_history: list,
    language: str = "en",
) -> dict:
    """
    Agent 1: Extract profile fields from user message, ask next question.

    Strategy (fixes the repeated-question bug):
    1. FIRST run pure-Python rule-based extraction (_rule_based_extract)
       → this reliably catches yes/no, numbers, states, names
    2. THEN call the LLM only to generate a friendly response / next question
       → LLM is NOT trusted to extract data, only to phrase the reply
    3. Merge rule-based updates into profile BEFORE asking next question
    4. Determine next_field from the UPDATED profile, not the old one
    """
   
    field_being_asked = _get_next_field(current_profile)
    rule_updates = _rule_based_extract(user_message, current_profile, field_being_asked)

    
    merged = {**current_profile}
    for k, v in rule_updates.items():
        if v is not None:
            merged[k] = v

    
    if merged.get("occupation") == "farmer" and merged.get("is_farmer") is None:
        merged["is_farmer"] = True

    
    just_answered = [k for k in rule_updates if rule_updates[k] is not None]
   
    next_field = _get_next_field(merged)

    filled = sum(1 for f in PROFILE_FIELDS if merged.get(f) is not None)
    pct = int((filled / len(PROFILE_FIELDS)) * 100)

    
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["en"])

    if next_field is None:
        
        if language == "hi":
            next_question = "शानदार! 🎉 आपकी सारी जानकारी मिल गई। अब 'परिणाम देखें' बटन दबाएं!"
        elif language == "bn":
            next_question = "দারুণ! 🎉 আপনার সমস্ত তথ্য পাওয়া গেছে। এখন 'ফলাফল দেখুন' বোতামটি চাপুন!"
        else:
            next_question = "Excellent! 🎉 I have all the information I need. Click 'View Results →' to see your eligible schemes!"
    else:
       
        try:
            system = f"""You are a friendly Indian government scheme advisor.
{lang_instruction}
The user just answered a question. Acknowledge their answer briefly (1 short sentence),
then ask the NEXT question listed below.
Keep it warm and simple. Maximum 2 sentences total.
Do NOT ask for information that is already filled."""

            already_filled = {k: v for k, v in merged.items() if v is not None}
            prompt = f"""User said: "{user_message}"
Fields just extracted from their answer: {just_answered}
Already known profile: {json.dumps(already_filled)}
NEXT question to ask (ask this exactly, translated to {language}): "{_get_question(next_field, language)}"

Reply with: brief acknowledgement + that next question. Nothing else."""

            llm_reply = call_llm(prompt, system, max_tokens=150)

        
            if llm_reply and len(llm_reply.strip()) > 5:
                next_question = llm_reply.strip()
            else:
                next_question = _get_question(next_field, language)

        except Exception as e:
            print(f"⚠️  LLM question generation failed: {e}")
            next_question = _get_question(next_field, language)

    return {
        "updated_profile": merged,
        "next_question": next_question,
        "profile_complete_pct": pct,
        "language_detected": language,
    }


def _fallback_next_question(profile: dict, language: str) -> str:
    """Deterministic fallback — returns next unanswered question."""
    for field in PROFILE_FIELDS:
        if profile.get(field) is None:
            return _get_question(field, language)
    if language == "hi":
        return "आपकी जानकारी पूरी हो गई है! 'परिणाम देखें' बटन दबाएं।"
    if language == "bn":
        return "আপনার তথ্য সম্পূর্ণ হয়েছে! 'ফলাফল দেখুন' বোতামটি চাপুন।"
    return "Your profile is complete! Click 'View Results →'."



def check_eligibility(profile: dict, schemes: list) -> list:
    """
    Agent 2: Pure rule-based eligibility check.
    Returns ranked list: ELIGIBLE → PARTIAL → NOT_ELIGIBLE
    """
    results = []

    age = profile.get("age") or 0
    gender = (profile.get("gender") or "any").lower()
    state = (profile.get("state") or "").strip()
    occupation = (profile.get("occupation") or "").lower()
    income = profile.get("income_annual_inr") or 999999999
    caste = (profile.get("caste_category") or "General").upper()
    is_bpl = bool(profile.get("is_bpl"))
    has_disability = bool(profile.get("has_disability"))
    is_farmer = bool(profile.get("is_farmer")) or occupation == "farmer"
    number_of_children = profile.get("number_of_children") or 0

    for scheme in schemes:
        passed = []
        failed = []

        min_age = scheme.get("min_age") or 0
        max_age = scheme.get("max_age") or 999
        if min_age <= age <= max_age:
            passed.append(f"Age {age} is within {min_age}–{max_age}")
        else:
            failed.append(f"Age {age} not in required range {min_age}–{max_age}")

        req_gender = (scheme.get("gender") or "any").lower()
        if req_gender == "any" or req_gender == gender:
            passed.append("Gender matches")
        else:
            failed.append(f"Scheme is for {req_gender} only")

        scheme_state = (scheme.get("state") or "all").strip()
        if scheme_state.lower() == "all":
            passed.append("Available in all states")
        elif scheme_state.lower() == state.lower():
            passed.append(f"Available in {state}")
        else:
            failed.append(f"Only for {scheme_state} residents")

        income_limit = scheme.get("income_limit") or 999999999
        if income_limit >= 999999998:
            passed.append("No income limit for this scheme")
        elif income <= income_limit:
            passed.append(f"Income ₹{income:,.0f} within limit ₹{income_limit:,.0f}")
        else:
            failed.append(f"Income ₹{income:,.0f} exceeds limit ₹{income_limit:,.0f}")

        raw_caste_list = scheme.get("caste_list") or "[]"
        try:
            caste_list = json.loads(raw_caste_list) if isinstance(raw_caste_list, str) else raw_caste_list
        except Exception:
            caste_list = ["SC", "ST", "OBC", "General"]

        if not caste_list or caste in caste_list or "General" in caste_list:
            passed.append("Caste category eligible")
        else:
            failed.append(f"Caste {caste} not in eligible list: {caste_list}")

        bpl_required = bool(scheme.get("bpl_required"))
        if bpl_required and not is_bpl:
            failed.append("BPL card required but you don't have one")
        elif bpl_required and is_bpl:
            passed.append("BPL card holder — eligible")
        else:
            passed.append("BPL not required")

        disability_req = bool(scheme.get("disability_req"))
        if disability_req and not has_disability:
            failed.append("Disability certificate required")
        else:
            passed.append("Disability criteria met")

        raw_occ_list = scheme.get("occupation_list") or "[]"
        try:
            occ_list = json.loads(raw_occ_list) if isinstance(raw_occ_list, str) else raw_occ_list
        except Exception:
            occ_list = []

        if not occ_list:
            passed.append("Open to all occupations")
        elif occupation in occ_list:
            passed.append(f"Occupation '{occupation}' is eligible")
        elif is_farmer and "farmer" in occ_list:
            passed.append("Farmer occupation matches")
        else:
            failed.append(f"Occupation '{occupation}' not in: {occ_list}")

        if scheme.get("id") == "sukanya-samridhi":
            if number_of_children > 0:
                passed.append("Has children (girl child may be eligible)")
            else:
                failed.append("No children — need a girl child aged 0–10")

        n_failed = len(failed)
        if n_failed == 0:
            status = "ELIGIBLE"
        elif n_failed == 1:
            status = "PARTIAL"
        else:
            status = "NOT_ELIGIBLE"

        results.append({
            "scheme_id": scheme["id"],
            "scheme_name": scheme["name_en"],
            "scheme_name_hi": scheme.get("name_hi", ""),
            "scheme_name_bn": scheme.get("name_bn", ""),
            "status": status,
            "passed_criteria": passed,
            "failed_criteria": failed,
            "benefit_summary": scheme.get("benefit_summary", ""),
            "benefit_amount": scheme.get("benefit_amount", 0),
            "benefit_type": scheme.get("benefit_type", ""),
            "apply_url": scheme.get("apply_url", ""),
            "helpline": scheme.get("helpline", ""),
            "ministry": scheme.get("ministry", ""),
            "documents": json.loads(scheme.get("documents") or "[]"),
        })

    order = {"ELIGIBLE": 0, "PARTIAL": 1, "NOT_ELIGIBLE": 2}
    results.sort(key=lambda x: order.get(x["status"], 3))
    return results

def generate_guide(scheme: dict, profile: dict, language: str = "en") -> str:
    """Agent 3: Generate personalized step-by-step application guide using Groq."""
    lang_instruction = LANGUAGE_INSTRUCTIONS.get(language, LANGUAGE_INSTRUCTIONS["en"])

    profile_summary = (
        f"Age: {profile.get('age', 'unknown')}, "
        f"Gender: {profile.get('gender', 'unknown')}, "
        f"State: {profile.get('state', 'unknown')}, "
        f"Occupation: {profile.get('occupation', 'unknown')}, "
        f"Caste: {profile.get('caste_category', 'unknown')}, "
        f"BPL: {'Yes' if profile.get('is_bpl') else 'No'}, "
        f"Annual Income: ₹{profile.get('income_annual_inr', 'unknown')}"
    )

    docs = scheme.get("documents", "[]")
    if isinstance(docs, str):
        docs = json.loads(docs or "[]")

    system = f"""You are an expert who helps Indian citizens apply for government welfare schemes.
{lang_instruction}
Generate a clear, numbered step-by-step application guide.
Keep it simple — many users have limited literacy.
Maximum 350 words. Use numbered lists only."""

    prompt = f"""Scheme: {scheme.get('name_en')}
Ministry: {scheme.get('ministry')}
Benefit: {scheme.get('benefit_summary')}
Required Documents: {docs}
Apply URL: {scheme.get('apply_url', 'Not available online')}
Helpline: {scheme.get('helpline', 'Not available')}
Processing Time: {scheme.get('processing_days', '?')} days
Deadline: {scheme.get('deadline', 'Open')}
Common Mistakes: {scheme.get('rejection_tips', 'None specified')}
Applicant profile: {profile_summary}

Write:
1. How to apply (step-by-step, online or offline)
2. Documents needed (only what's relevant for this applicant)
3. Common mistakes to avoid
4. How long it takes to receive benefit
5. Helpline number

Language: {language}"""

    result = call_llm(prompt, system, max_tokens=800)
    if not result:
        return _fallback_guide(scheme, language)
    return result


def _fallback_guide(scheme: dict, language: str) -> str:
    docs = scheme.get("documents", "[]")
    if isinstance(docs, str):
        docs = json.loads(docs or "[]")
    docs_str = "\n".join(f"   - {d}" for d in docs)
    return f"""Application Guide for {scheme.get('name_en', 'this scheme')}

Benefit: {scheme.get('benefit_summary', '')}

Steps to Apply:
1. Collect all required documents
2. Visit the official website: {scheme.get('apply_url', 'Check myscheme.gov.in')}
3. Fill the application form with your details
4. Upload or submit your documents
5. Note your application reference number
6. Track your application status online

Required Documents:
{docs_str}

Helpline: {scheme.get('helpline', '1800-111-555')}
Expected Processing: {scheme.get('processing_days', '30-60')} days

For more help, visit your nearest Common Service Centre (CSC) or call the helpline."""


def translate_text(text: str, target_language: str) -> str:
    """Agent 4: Translate English text to Hindi or Bengali using Gemini Flash."""
    if target_language == "en" or not text:
        return text

    lang_name = "Hindi" if target_language == "hi" else "Bengali"
    system = "You are a professional translator specializing in Indian government communications."
    prompt = (
        f"Translate the following text to {lang_name}. "
        f"Keep government scheme names in English. "
        f"Output ONLY the translation:\n\n{text}"
    )

    gemini_key = os.getenv("GEMINI_API_KEY", "")
    if gemini_key and gemini_key != "your_gemini_api_key_here":
        try:
            import google.generativeai as genai
            genai.configure(api_key=gemini_key)
            model = genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content(f"{system}\n\n{prompt}")
            return response.text
        except Exception as e:
            print(f"⚠️  Translation failed: {e}")

    result = call_llm(prompt, system, max_tokens=600)
    return result if result else text