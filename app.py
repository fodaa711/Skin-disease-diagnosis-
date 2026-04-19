import streamlit as st
from fpdf import FPDF
from pipeline import chat_stream, log_conversation, KNOWN_DISEASES_EN, KNOWN_DISEASES_AR

st.set_page_config(
    page_title="DermaCam",
    page_icon="🩺",
    layout="centered"
)

REFUSAL_KEYWORDS = [
    "not authorized", "not in my current database",
    "specialize only", "cannot answer",
    "غير مخوّل", "ليس ضمن قاعدة", "متخصص فقط",
]

GREETING_KEYWORDS = [
    "nice to meet you", "great to have", "how can i assist",
    "how's your day", "i'm doing well", "i'm here to help",
    "great to chat", "happy to help", "how can i help",
    "كيف حالك", "أنا بخير", "يسعدني", "كيف يمكنني",
]

# ── Sidebar ────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🩺 DermaCam")
    st.markdown("*AI skin disease assistant*")
    st.markdown("**English & العربية**")
    st.divider()

    st.markdown("### 📋 Supported Diseases")
    for en, ar in zip(KNOWN_DISEASES_EN, KNOWN_DISEASES_AR):
        st.markdown(f"• {en} / {ar}")
    st.divider()

if st.button("📄 Export chat as PDF", use_container_width=True):
    if st.session_state.get("messages"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_auto_page_break(auto=True, margin=15)
        pdf.cell(200, 10, "DermaCam Chat Export", ln=True, align="C")
        pdf.cell(200, 4, "", ln=True)
        for msg in st.session_state.messages:
            role = "You" if msg["role"] == "user" else "DermaCam"
            text = f"{role}: {msg['content']}"
            text = text.encode("latin-1", "replace").decode("latin-1")
            pdf.multi_cell(0, 8, text)
            pdf.cell(200, 4, "", ln=True)
        pdf_bytes = bytes(pdf.output())  # ← fixed line
        st.download_button(
            label="⬇️ Download PDF",
            data=pdf_bytes,
            file_name="dermacam_chat.pdf",
            mime="application/pdf",
            use_container_width=True
        )
    else:
        st.warning("No chat to export yet!")

    if st.button("🗑️ Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history  = []
        st.rerun()

    st.divider()
    st.caption("⚠️ For educational purposes only. Always consult a licensed dermatologist.")

# ── Main chat area ─────────────────────────────────────────────────────────
st.title("🩺 DermaCam")
st.caption("Skin disease assistant — English & العربية")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# Welcome message on first load only
if not st.session_state.messages:
    with st.chat_message("assistant"):
        st.markdown(
            "👋 Hello! I'm **DermaCam**, your AI assistant for skin diseases.\n\n"
            "I can help you learn about **8 skin conditions** in both **English and Arabic**.\n\n"
            "Ask me about symptoms, causes, severity, or when to see a doctor. What would you like to know?"
        )

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("badge"):
            st.caption(msg["badge"])

# ── User input ─────────────────────────────────────────────────────────────
query = st.chat_input("Ask about a skin condition... / اسأل عن حالة جلدية...")

if query:
    with st.chat_message("user"):
        st.markdown(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_answer = ""

        stream, meta = chat_stream(query, history=st.session_state.history)

        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                full_answer += delta
                response_placeholder.markdown(full_answer + "▌")

        response_placeholder.markdown(full_answer)
        log_conversation(query, full_answer, meta["lang"], meta["score"])

        # Confidence badge
        badge_text  = None
        is_refusal  = any(kw.lower() in full_answer.lower() for kw in REFUSAL_KEYWORDS)
        is_greeting = any(kw.lower() in full_answer.lower() for kw in GREETING_KEYWORDS)

        show_source = (
            not meta["fallback"]
            and meta["chunks"]
            and not is_refusal
            and not is_greeting
            and meta["score"] >= 0.55
        )

        if show_source:
            top     = meta["chunks"][0]
            lang    = meta["lang"]
            disease = top["disease_ar"] if lang == "ar" else top["disease_en"]
            section = top["section"].replace("_", " ").title()
            score   = round(meta["score"] * 100)

            if score >= 75:
                badge = "🟢 High confidence"
            elif score >= 55:
                badge = "🟡 Medium confidence"
            else:
                badge = "🔴 Low confidence"

            badge_text = f"{badge} | 📚 {disease} — {section} ({score}%)"
            st.caption(badge_text)

    st.session_state.messages.append({
        "role":    "assistant",
        "content": full_answer,
        "badge":   badge_text,
    })

    st.session_state.history.append({"role": "user",      "content": query})
    st.session_state.history.append({"role": "assistant", "content": full_answer})

    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]