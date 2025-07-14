# app.py
import streamlit as st
import torch, json, requests
from transformers import pipeline

# ------------------------------------------------------------------
# 0. Config
# ------------------------------------------------------------------
st.set_page_config(page_title="🎙️ Whisper → Emotion → AI", layout="wide")
st.title("🎙️ Whisper → Emotion → AI Assistant")

OPENROUTER_KEY = st.secrets["OPENROUTER_KEY"]
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# ------------------------------------------------------------------
# 1. Cached ML models (no ffmpeg)
# ------------------------------------------------------------------
@st.cache_resource
def load_asr():
    return pipeline(
        "automatic-speech-recognition",
        model="openai/whisper-base",
        device=0 if torch.cuda.is_available() else -1,
    )

@st.cache_resource
def load_emotion():
    return pipeline(
        "text-classification",
        model="cardiffnlp/twitter-roberta-base-emotion",
        return_all_scores=False,
        device=0 if torch.cuda.is_available() else -1,
    )

asr_pipe = load_asr()
emo_pipe = load_emotion()

# ------------------------------------------------------------------
# 2. Small helpers for OpenRouter
# ------------------------------------------------------------------
def llm(messages, model="openai/gpt-3.5-turbo", max_tokens=512):
    headers = {
        "Authorization": f"Bearer {OPENROUTER_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    r = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

# ------------------------------------------------------------------
# 3. Sidebar – audio input
# ------------------------------------------------------------------
audio_file = st.sidebar.file_uploader("Upload audio (wav/mp3)", type=["wav", "mp3"])
mic_audio = st.sidebar.audio_input("🎤 Record a snippet")

audio_bytes = mic_audio or audio_file

if not audio_bytes:
    st.stop()

# ------------------------------------------------------------------
# 4. ASR + Emotion
# ------------------------------------------------------------------
with st.spinner("Transcribing…"):
    transcript = asr_pipe(audio_bytes)["text"].strip()

with st.spinner("Detecting emotion…"):
    emo = emo_pipe(transcript)[0]
    emotion_label = emo["label"].upper()
    emotion_score = emo["score"]

st.success(f"**Transcript:** {transcript}")
st.metric("Detected Emotion", emotion_label, delta=f"{emotion_score:.0%}")

st.divider()

# ------------------------------------------------------------------
# 5. AI-powered extras via tabs
# ------------------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["💬 Chat with transcript", "🧠 Emotion explanation", "🌐 Translate", "✍️ Rewrite tone"]
)

with tab1:
    st.subheader("💬 Ask anything about the transcript")
    question = st.text_input("Your question", placeholder="Summarize this in 2 sentences")
    if st.button("Ask", key="ask"):
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"The transcript is:\n{transcript}\n\n{question}"},
        ]
        with st.spinner("Thinking…"):
            answer = llm(messages)
        st.markdown(answer)

with tab2:
    st.subheader("🧠 Why this emotion was detected")
    prompt = (
        f"The following text was classified as **{emotion_label}** ({emotion_score:.0%} confident).\n"
        f"Explain briefly why this tone makes sense:\n\n{transcript}"
    )
    if st.button("Explain", key="explain"):
        with st.spinner("Generating explanation…"):
            exp = llm([{"role": "user", "content": prompt}])
        st.markdown(exp)

with tab3:
    st.subheader("🌐 Translate transcript")
    target = st.text_input("Target language", "Spanish")
    if st.button("Translate", key="translate"):
        prompt = f"Translate the following text into {target}:\n\n{transcript}"
        with st.spinner("Translating…"):
            tr = llm([{"role": "user", "content": prompt}])
        st.markdown(tr)

with tab4:
    st.subheader("✍️ Rewrite in another tone")
    tone = st.selectbox("Choose tone", ["polite", "excited", "formal", "concise"])
    if st.button("Rewrite", key="rewrite"):
        prompt = f"Re-write the following message in a {tone} tone:\n\n{transcript}"
        with st.spinner("Rewriting…"):
            rw = llm([{"role": "user", "content": prompt}])
        st.markdown(rw)