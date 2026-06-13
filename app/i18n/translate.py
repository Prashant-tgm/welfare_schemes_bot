"""
i18n & Code-Mixing Layer
========================
- Static UI strings for Hindi/Tamil/English.
- Code-mixing glossary catches common Hinglish/Tanglish govt terms before
  any MT step (handles "मेरा aadhaar खो गया" etc.)
- IndicTrans2 wrapper for translating free-text RAG answers, if needed.
"""

GLOSSARY = {
    "aadhar": "Aadhaar Card", "aadhaar": "Aadhaar Card", "आधार": "Aadhaar Card",
    "ration card": "Ration Card", "राशन कार्ड": "Ration Card", "राशन": "Ration Card",
    "bank account": "Bank Account", "khata": "Bank Account", "खाता": "Bank Account",
    "job card": "MGNREGA Job Card", "नरेगा": "MGNREGA Job Card", "मनरेगा": "MGNREGA Job Card",
    "pucca house": "Pucca House", "pucca ghar": "Pucca House", "पक्का मकान": "Pucca House",
    "bpl card": "BPL Card", "बीपीएल": "BPL Card",
    "gas connection": "LPG Gas Connection", "गैस कनेक्शन": "LPG Gas Connection",
    "kisan": "Farmer", "किसान": "Farmer",
    "vidhwa": "Widow", "विधवा": "Widow",
}

LANG_CODES = {
    "hindi": "hin_Deva", "tamil": "tam_Taml", "english": "eng_Latn",
    "1": "hin_Deva", "2": "tam_Taml", "3": "eng_Latn",
}

LANG_NAMES = {"hin_Deva": "Hindi", "tam_Taml": "Tamil", "eng_Latn": "English"}

_translator_cache = {}


def normalize_codemixed(text: str) -> str:
    """Replace common code-mixed/Hinglish welfare terms with canonical English terms."""
    lowered = text.lower()
    normalized = text
    for term, canonical in GLOSSARY.items():
        if term.lower() in lowered:
            normalized += f" [{canonical}]"
    return normalized


def get_translator(src_lang, tgt_lang):
    key = (src_lang, tgt_lang)
    if key in _translator_cache:
        return _translator_cache[key]

    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if src_lang == "eng_Latn":
        model_name = "ai4bharat/indictrans2-en-indic-1B"
    else:
        model_name = "ai4bharat/indictrans2-indic-en-1B"

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, trust_remote_code=True)
    _translator_cache[key] = (tokenizer, model)
    return tokenizer, model


def translate(text, src_lang, tgt_lang):
    if src_lang == tgt_lang:
        return text

    import torch
    tokenizer, model = get_translator(src_lang, tgt_lang)

    prefixed = f"{src_lang} {tgt_lang} {text}"
    inputs = tokenizer(prefixed, return_tensors="pt", padding=True, truncation=True, max_length=256)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=256, num_beams=5)
    return tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]


# ---- Static UI strings ----
STRINGS = {
    "welcome": {
        "hin_Deva": "नमस्ते! मैं आपकी सरकारी योजनाओं में मदद करूँगा। भाषा चुनें: 1. हिंदी 2. தமிழ் 3. English",
        "tam_Taml": "வணக்கம்! அரசு திட்டங்களைப் பற்றி நான் உதவுவேன். மொழியைத் தேர்வு செய்யவும்: 1. हिंदी 2. தமிழ் 3. English",
        "eng_Latn": "Welcome! I'll help you find government welfare schemes. Choose language: 1. Hindi 2. Tamil 3. English",
    },
    "ask_occupation": {
        "hin_Deva": "आपका मुख्य काम क्या है?\n1) किसान\n2) मज़दूर\n3) स्वरोजगार/दुकानदार\n4) बेरोज़गार\n5) अन्य",
        "tam_Taml": "உங்கள் முதன்மை தொழில் என்ன?\n1) விவசாயி\n2) தொழிலாளர்\n3) சுயதொழில்/கடைக்காரர்\n4) வேலையில்லாதவர்\n5) மற்றவை",
        "eng_Latn": "What is your main occupation?\n1) Farmer\n2) Laborer\n3) Self-employed/Shopkeeper\n4) Unemployed\n5) Other",
    },
    "ask_age_gender": {
        "hin_Deva": "आपकी उम्र और लिंग बताएं (जैसे: 35 महिला)",
        "tam_Taml": "உங்கள் வயது மற்றும் பாலினம் கூறவும் (எ.கா: 35 பெண்)",
        "eng_Latn": "Please tell your age and gender (e.g. 35 female)",
    },
    "ask_income": {
        "hin_Deva": "आपकी सालाना पारिवारिक आय कितनी है?\n1) ₹1 लाख से कम\n2) ₹1-2.5 लाख\n3) ₹2.5 लाख से ज़्यादा",
        "tam_Taml": "உங்கள் ஆண்டு குடும்ப வருமானம் என்ன?\n1) ₹1 லட்சத்திற்கும் குறைவு\n2) ₹1-2.5 லட்சம்\n3) ₹2.5 லட்சத்திற்கும் மேல்",
        "eng_Latn": "What is your annual family income?\n1) Below ₹1 lakh\n2) ₹1-2.5 lakh\n3) Above ₹2.5 lakh",
    },
    "ask_land_house": {
        "hin_Deva": "क्या आपके पास खेती की ज़मीन है? और क्या आपका पक्का मकान है? (हाँ/नहीं, हाँ/नहीं)",
        "tam_Taml": "உங்களுக்கு விவசாய நிலம் உள்ளதா? பக்கா வீடு உள்ளதா? (ஆம்/இல்லை, ஆம்/இல்லை)",
        "eng_Latn": "Do you own farmland? Do you have a pucca (concrete) house? (yes/no, yes/no)",
    },
    "ask_special": {
        "hin_Deva": "क्या आप इनमें से कोई हैं? (नंबर भेजें, कई चुन सकते हैं)\n1) बीपीएल कार्ड धारक\n2) विधवा\n3) 10 साल से छोटी बेटी है\n4) स्ट्रीट वेंडर\n5) इनमें से कोई नहीं",
        "tam_Taml": "நீங்கள் இவற்றில் ஏதேனும் ஒன்றா? (எண்ணை அனுப்பவும், பல தேர்வு செய்யலாம்)\n1) பிபிஎல் கார்டு\n2) விதவை\n3) 10 வயதுக்குட்பட்ட மகள்\n4) தெரு வியாபாரி\n5) இவை எதுவுமில்லை",
        "eng_Latn": "Do any of these apply to you? (send numbers, can pick multiple)\n1) BPL card holder\n2) Widow\n3) Have a daughter under 10\n4) Street vendor\n5) None of these",
    },
    "shortlist_header": {
        "hin_Deva": "आपके लिए ये योजनाएं उपलब्ध हैं:",
        "tam_Taml": "உங்களுக்கான திட்டங்கள்:",
        "eng_Latn": "Schemes you may be eligible for:",
    },
    "checklist_header": {
        "hin_Deva": "दस्तावेज़ सूची (इसे सेव करें):",
        "tam_Taml": "ஆவண பட்டியல் (இதை சேமிக்கவும்):",
        "eng_Latn": "Document checklist (save this):",
    },
    "no_match": {
        "hin_Deva": "माफ़ कीजिए, दी गई जानकारी से कोई योजना मेल नहीं खाई। कृपया नज़दीकी CSC केंद्र से संपर्क करें।",
        "tam_Taml": "மன்னிக்கவும், கொடுக்கப்பட்ட தகவலின்படி எந்த திட்டமும் பொருந்தவில்லை. அருகிலுள்ள CSC மையத்தைத் தொடர்பு கொள்ளவும்.",
        "eng_Latn": "Sorry, no schemes matched based on the info given (or we don't have verified rules yet). Please visit your nearest Common Service Centre (CSC).",
    },
    "satisfaction_check": {
        "hin_Deva": "\nक्या इससे आपके सवाल का जवाब मिल गया? (1=हाँ 2=नहीं 3=व्यक्ति से बात करें)",
        "tam_Taml": "\nஇது உங்கள் கேள்விக்கு பதிலளித்ததா? (1=ஆம் 2=இல்லை 3=ஒரு நபருடன் பேசவும்)",
        "eng_Latn": "\nDid this answer your question? (1=Yes 2=No 3=Talk to a person)",
    },
    "ask_again": {
        "hin_Deva": "माफ़ कीजिए। कृपया अपना सवाल अलग तरीके से बताएं, या और जानकारी दें।",
        "tam_Taml": "மன்னிக்கவும். உங்கள் கேள்வியை வேறு வழியில் கேட்கவும், அல்லது கூடுதல் தகவல் கொடுக்கவும்.",
        "eng_Latn": "Sorry about that. Please try rephrasing your question, or give more details.",
    },
    "handoff": {
        "hin_Deva": "ठीक है। कृपया अपने नज़दीकी Common Service Centre (CSC) पर जाएं या 1800-XXX-XXXX पर कॉल करें। वहां एक व्यक्ति आपकी मदद करेगा।",
        "tam_Taml": "சரி. அருகிலுள்ள Common Service Centre (CSC) க்குச் செல்லவும் அல்லது 1800-XXX-XXXX ஐ அழைக்கவும். அங்கு ஒரு நபர் உங்களுக்கு உதவுவார்.",
        "eng_Latn": "Okay. Please visit your nearest Common Service Centre (CSC) or call 1800-XXX-XXXX. A person there can help you further.",
    },
    "max_turns_handoff": {
        "hin_Deva": "मुझे लगता है कि आपको और मदद की ज़रूरत है। कृपया अपने नज़दीकी CSC केंद्र पर जाएं जहाँ एक व्यक्ति आपकी पूरी मदद कर सकता है।",
        "tam_Taml": "உங்களுக்கு கூடுதல் உதவி தேவை என்று நினைக்கிறேன். அருகிலுள்ள CSC மையத்திற்குச் செல்லவும், அங்கு ஒரு நபர் முழுமையாக உதவ முடியும்.",
        "eng_Latn": "It looks like you need more detailed help than I can give over chat. Please visit your nearest CSC centre, where a person can assist you fully.",
    },
    "goodbye": {
        "hin_Deva": "धन्यवाद! शुभकामनाएं। फिर से बात करने के लिए 'hi' लिखें।",
        "tam_Taml": "நன்றி! வாழ்த்துகள். மீண்டும் பேச 'hi' என தட்டச்சு செய்யவும்.",
        "eng_Latn": "Thank you! Good luck with your application. Type 'hi' to start a new conversation anytime.",
    },
}


def t(key, lang_code):
    return STRINGS.get(key, {}).get(lang_code, STRINGS.get(key, {}).get("eng_Latn", key))
