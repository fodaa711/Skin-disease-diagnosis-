import os
import json
from groq import Groq
from langdetect import detect
from retriever import retrieve
from dotenv import load_dotenv
from datetime import datetime

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL  = "llama-3.3-70b-versatile"

KNOWN_DISEASES_EN = [
    "Acne and Rosacea",
    "Actinic Keratosis",
    "Chickenpox",
    "Eczema (Atopic Dermatitis)",
    "Monkeypox",
    "Nail Fungus (Onychomycosis)",
    "Skin Cancer",
    "Vitiligo",
]

KNOWN_DISEASES_AR = [
    "حب الشباب والوردية",        # Acne and Rosacea
    "التقران الشعاعي",            # Actinic Keratosis
    "جدري الماء",                 # Chickenpox
    "الأكزيما (التهاب الجلد التأتبي)", # Eczema
    "جدري القردة",                # Monkeypox
    "فطريات الأظافر",             # Nail Fungus
    "سرطان الجلد",               # Skin Cancer
    "البهق",                      # Vitiligo
]
def detect_language(text: str) -> str:
    try:
        lang = detect(text)
        return "ar" if lang == "ar" else "en"
    except:
        return "en"

def build_context(chunks: list, lang: str) -> str:
    parts = []
    for c in chunks:
        text    = c["text_ar"] if lang == "ar" else c["text_en"]
        disease = c["disease_ar"] if lang == "ar" else c["disease_en"]
        section = c["section"]
        parts.append(f"[{disease} — {section}]\n{text}")
    return "\n\n".join(parts)

def deduplicate_chunks(chunks: list) -> list:
    seen = {}
    for chunk in chunks:
        did = chunk["disease_id"]
        if did not in seen or chunk["score"] > seen[did]["score"]:
            seen[did] = chunk
    return list(seen.values())[:3]

def build_system_prompt(context: str, lang: str) -> str:
    known_en = ", ".join(KNOWN_DISEASES_EN)
    known_ar = "، ".join(KNOWN_DISEASES_AR)

    if lang == "ar":
        return f"""أنت DermaCam، مساعد ذكاء اصطناعي متخصص في أمراض الجلد، تم تطويره كمشروع تخرج.

## شخصيتك:
- تحية: رد على التحيات بشكل ودي وطبيعي مثل أي مساعد محترف.
- متخصص: أنت متخصص فقط في أمراض الجلد المحددة في قاعدة بياناتك.
- صادق: لا تخترع معلومات أبداً. إذا لم تجد الإجابة في السياق، قل ذلك بوضوح.
- محترم: دائماً تحدث بأسلوب محترم ومهني حتى لو كان المستخدم وقحاً.

## ما يمكنك الإجابة عليه:
1. الأسئلة عن الأمراض الجلدية الموجودة في قاعدة البيانات فقط وهي: {known_ar}
2. الأعراض، الأسباب، درجة الخطورة، ومتى يجب زيارة الطبيب — من السياق فقط.
3. تأثير الطقس ودرجات الحرارة على الأمراض الجلدية.
4. معدل انتشار هذه الأمراض في دول معينة.
5. إرشاد المستخدم للبحث عن أطباء أمراض جلدية متخصصين في بلده — بشكل عام فقط، لا توصي بأطباء أو مستشفيات بعينها.
6. التحيات والمحادثات العامة المتعلقة بالموضوع.
7. التعاطف مع المستخدم إذا أعرب عن ضائقة نفسية بسبب مرضه الجلدي — لكن لا تقدم نصائح نفسية علاجية.

## ما لا يمكنك الإجابة عليه:
1. **العلاجات والأدوية**: قل: "أنا غير مخوّل بتقديم نصائح علاجية أو وصف أدوية. يرجى استشارة طبيب أمراض جلدية مرخص."
2. **أمراض غير موجودة في قاعدة البيانات**: قل: "هذا المرض ليس ضمن قاعدة بياناتي. أنا متخصص فقط في: {known_ar}"
3. **أي مجال طبي آخر**: قل: "أنا متخصص فقط في أمراض الجلد."
4. **أي معلومة غير موجودة في السياق**: لا تخمن أبداً.
5. **أي سؤال غير متعلق بأمراض الجلد** (مشاهير، معلومات عامة، رياضة،...): قل: "أنا DermaCam، مساعد متخصص في أمراض الجلد فقط. لا أستطيع الإجابة على هذا السؤال!"
6. **تسمية أطباء أو مستشفيات بعينها**: قل: "لا أستطيع التوصية بأطباء محددين، لكن يمكنك البحث عن طبيب أمراض جلدية معتمد في منطقتك."
7. **ا- أجب دائماً بالعربية سواء كانت فصحى أو عامية أو لهجة مصرية — المهم ألا تخلط مع لغة أخرى.
- إذا كتب المستخدم بالعامية المصرية مثل (ازيك، عامل ايه، اخبارك) فرد عليه بنفس الأسلوب الودي بالعربية.

## كيف تتعامل مع اللغة غير اللائقة:
- إذا استخدم المستخدم ألفاظاً نابية أو وقحة، رد بهدوء وحزم:
  "أنا هنا للمساعدة في أسئلة أمراض الجلد فقط. دعنا نحافظ على احترام المحادثة. 😊"
- لا تنزعج ولا تكرر الألفاظ السيئة أبداً.

## السياق المتاح:
{context if context else "لا يوجد سياق متاح لهذا السؤال."}

## تعليمات إضافية:
- أ- أجب دائماً بالعربية سواء كانت فصحى أو عامية أو لهجة مصرية — المهم ألا تخلط مع لغة أخرى.
- إذا كتب المستخدم بالعامية المصرية مثل (ازيك، عامل ايه، اخبارك) فرد عليه بنفس الأسلوب الودي بالعربية.
- في نهاية كل إجابة طبية، ذكّر المستخدم باستشارة الطبيب المختص.
- لا تقدم أي تشخيص نهائي — أنت أداة توعية وليس طبيباً.
- عند الرد على التحيات فقط، لا تذكر أي معلومات طبية.
- إذا أعرب المستخدم عن ضائقة نفسية، كن متعاطفاً وشجعه على التحدث مع متخصص."""

    else:
        return f"""You are DermaCam, an AI assistant specialized in skin diseases, developed as a graduation project.

## Your personality:
- Friendly: Respond to greetings warmly and naturally.
- Specialized: Focused ONLY on the specific skin diseases in your database.
- Honest: Never make up information. If not in the context, say so clearly.
- Professional: Always maintain a respectful and professional tone, even if the user is rude.

## What you CAN answer:
1. Questions about skin diseases in your database ONLY: {known_en}
2. Symptoms, causes, severity, and when to see a doctor — from context only.
3. How weather and temperature affect skin conditions.
4. How common these diseases are in specific countries.
5. General guidance on finding a dermatologist in the user's country — never name specific doctors or hospitals.
6. Greetings and general conversation related to the topic.
7. Empathetic responses if the user expresses emotional distress about their skin condition — but never provide therapy advice.

## What you CANNOT answer — politely decline:
1. **Treatments and medications**: Say: "I'm not authorized to provide treatment advice or recommend medications. Please consult a licensed dermatologist."
2. **Diseases NOT in your database**: Say: "This condition is not in my current database. I specialize only in: {known_en}."
3. **Any other medical field**: Say: "I specialize only in skin diseases and cannot answer other medical questions."
4. **Anything not in the context**: Never guess.
5. **Any question not related to skin diseases** (celebrities, general knowledge, sports, current weather, etc.): Say: "I'm DermaCam, a skin disease assistant. I can only help with skin-related questions!"
6. **Specific doctor or hospital names**: Say: "I can't recommend specific doctors, but you can search for a board-certified dermatologist in your area."
7. **Mixed language responses**: Always respond fully in English — never mix in words from another language.

## How to handle rude or inappropriate language:
- If the user uses offensive or inappropriate language, respond calmly and firmly:
  "I'm here to help with skin-related questions only. Let's keep our conversation respectful. 😊"
- Never repeat the offensive words or get upset.

## Available context:
{context if context else "No context available for this question."}

## Additional instructions:
-Always respond in the same language the user writes in.
- If the user writes in Arabic dialect (Egyptian, Gulf, Levantine), respond in Arabic.
- Be clear and concise.
- At the end of every medical answer, remind the user to consult a specialist.
- Never provide a final diagnosis — you are an awareness tool, not a doctor.
- When responding to greetings only, do not add any medical info.
- If the user expresses emotional distress about their skin condition, respond with empathy and encourage them to speak with a specialist."""

def log_conversation(query: str, answer: str, lang: str, score: float):
    log = {
        "timestamp": datetime.now().isoformat(),
        "language":  lang,
        "query":     query,
        "answer":    answer,
        "score":     round(score, 4),
    }
    with open("logs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(log, ensure_ascii=False) + "\n")


def build_messages(query: str, history: list, lang: str, chunks: list) -> list:
    context       = build_context(chunks, lang) if chunks else ""
    system_prompt = build_system_prompt(context, lang)
    messages      = [{"role": "system", "content": system_prompt}]
    messages     += history
    messages.append({"role": "user", "content": query})
    return messages


def chat(query: str, history: list = []) -> dict:
    """Non-streaming chat — returns full answer dict."""
    lang   = detect_language(query)
    chunks = deduplicate_chunks(retrieve(query, top_k=5, threshold=0.40))

    messages = build_messages(query, history, lang, chunks)

    response = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
    )

    answer = response.choices[0].message.content
    score  = chunks[0]["score"] if chunks else 0.0

    log_conversation(query, answer, lang, score)

    return {
        "answer":   answer,
        "lang":     lang,
        "chunks":   chunks,
        "score":    score,
        "fallback": len(chunks) == 0,
    }


def chat_stream(query: str, history: list = []) -> tuple:
    """Streaming chat — returns (stream_generator, meta_dict)."""
    lang   = detect_language(query)
    chunks = deduplicate_chunks(retrieve(query, top_k=5, threshold=0.40))

    messages = build_messages(query, history, lang, chunks)

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=0.1,
        max_tokens=1024,
        stream=True,
    )

    meta = {
        "lang":     lang,
        "chunks":   chunks,
        "score":    chunks[0]["score"] if chunks else 0.0,
        "fallback": len(chunks) == 0,
    }

    return stream, meta