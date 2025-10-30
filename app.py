import streamlit as st
import docx
import fitz  # PyMuPDF for PDFs
import io
import os
import re
import tempfile
from gtts import gTTS
import base64

# Optional dependencies
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


# -----------------------------------
# Utility Functions
# -----------------------------------
def extract_text(file):
    """Extract text from txt, docx, or pdf file."""
    name = file.name.lower()

    # Text files
    if name.endswith(".txt"):
        return file.read().decode("utf-8", errors="ignore")

    # DOCX
    elif name.endswith(".docx"):
        doc = docx.Document(file)
        return "\n".join([p.text for p in doc.paragraphs])

    # PDF
    elif name.endswith(".pdf"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file.read())
            tmp_path = tmp.name

        text = ""
        pdf = fitz.open(tmp_path)
        for page in pdf:
            text += page.get_text("text") or ""
        pdf.close()
        os.remove(tmp_path)
        return text

    else:
        raise ValueError("Unsupported file format. Please upload .txt, .docx, or .pdf")


def chunk_text(text, max_chars=3000):
    """Split text into smaller chunks for TTS or rewriting."""
    if not text:
        return []
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current = [], ""

    for sent in sentences:
        if len(current) + len(sent) < max_chars:
            current += sent + " "
        else:
            chunks.append(current.strip())
            current = sent + " "
    if current.strip():
        chunks.append(current.strip())
    return chunks


def rewrite_local(text):
    """Simple rewrite logic (local)."""
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.replace("AI", "Artificial Intelligence (AI)")
    sents = re.split(r'(?<=[.!?]) +', text)
    return " ".join([s.capitalize() for s in sents if s.strip()])


def rewrite_openai(text, model="gpt-4o-mini"):
    """Rewrite using OpenAI API."""
    if not OPENAI_AVAILABLE:
        raise RuntimeError("OpenAI not installed.")
    if not getattr(openai, "api_key", None):
        raise RuntimeError("OpenAI key not set.")

    prompt = f"Rewrite the following text naturally for audiobook narration:\n\n{text}"

    resp = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800,
    )
    return resp.choices[0].message.content.strip()


def generate_tts(chunks, lang="en"):
    """Convert chunks to speech and return list of mp3 paths."""
    mp3_files = []
    for i, chunk in enumerate(chunks, 1):
        if chunk.strip():
            tts = gTTS(chunk, lang=lang)
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=f"_{i}.mp3")
            tts.save(tmp.name)
            mp3_files.append(tmp.name)
    return mp3_files


def merge_mp3(mp3_paths, output_path):
    """Merge multiple mp3 files into one."""
    if PYDUB_AVAILABLE:
        combined = None
        for p in mp3_paths:
            seg = AudioSegment.from_file(p, format="mp3")
            combined = seg if combined is None else combined + seg
        combined.export(output_path, format="mp3")
    else:
        with open(output_path, "wb") as outfile:
            for p in mp3_paths:
                with open(p, "rb") as infile:
                    outfile.write(infile.read())
    return output_path


# -----------------------------------
# Streamlit UI
# -----------------------------------
st.set_page_config(page_title="📚 Audiobook Generator", layout="wide")
st.title("📖 Audiobook Rewriter & Generator")
st.write("Upload your `.txt`, `.docx`, or `.pdf` file to rewrite and convert it into an audiobook!")

# Sidebar settings
st.sidebar.header("⚙️ Settings")
rewrite_mode = st.sidebar.radio("Rewrite Method", ["Local (Simple)", "OpenAI (Advanced)"])
lang = st.sidebar.selectbox("Language for TTS", ["en", "hi", "en-in", "en-us"])
max_chars = st.sidebar.slider("Max Characters per Chunk", 1000, 8000, 3000, step=500)

# OpenAI API Key (secure)
api_key = None
if rewrite_mode == "OpenAI (Advanced)":
    if "OPENAI_API_KEY" in st.secrets:
        api_key = st.secrets["OPENAI_API_KEY"]
    elif os.getenv("OPENAI_API_KEY"):
        api_key = os.getenv("OPENAI_API_KEY")
    else:
        api_key = st.sidebar.text_input("Enter your OpenAI API Key", type="password")

    if api_key:
        openai.api_key = api_key

# File uploader
uploaded_file = st.file_uploader("📤 Upload a file", type=["txt", "docx", "pdf"])

if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name}")
    try:
        text = extract_text(uploaded_file)
    except Exception as e:
        st.error(f"Failed to extract text: {e}")
        st.stop()

    st.text_area("📜 Original Text Preview", text[:2000], height=250)

    if st.button("🔁 Rewrite Text"):
        with st.spinner("Rewriting in progress..."):
            chunks = chunk_text(text, max_chars=1500)
            rewritten = []

            for chunk in chunks:
                try:
                    if rewrite_mode == "OpenAI (Advanced)" and api_key:
                        rewritten_text = rewrite_openai(chunk)
                    else:
                        rewritten_text = rewrite_local(chunk)
                except Exception as e:
                    st.warning(f"Rewrite failed ({e}), using local method.")
                    rewritten_text = rewrite_local(chunk)
                rewritten.append(rewritten_text)

            final_text = "\n\n".join(rewritten)
            st.session_state["final_text"] = final_text

        st.success("✅ Rewriting complete!")
        st.text_area("📝 Rewritten Text", final_text[:4000], height=300)
        st.download_button("📥 Download Rewritten Text", data=final_text.encode("utf-8"),
                           file_name="rewritten_text.txt", mime="text/plain")

    if st.button("🎧 Generate Audio"):
        if "final_text" in st.session_state:
            text_to_read = st.session_state["final_text"]
        else:
            text_to_read = text

        chunks = chunk_text(text_to_read, max_chars=max_chars)
        with st.spinner("Generating audio..."):
            try:
                mp3_paths = generate_tts(chunks, lang=lang)
                final_mp3 = os.path.join(tempfile.gettempdir(), f"{uploaded_file.name.split('.')[0]}_audiobook.mp3")
                merge_mp3(mp3_paths, final_mp3)

                st.audio(final_mp3)
                with open(final_mp3, "rb") as f:
                    st.download_button("📥 Download MP3", data=f.read(),
                                       file_name=os.path.basename(final_mp3),
                                       mime="audio/mpeg")
            except Exception as e:
                st.error(f"TTS generation failed: {e}")
else:
    st.info("Please upload a file to begin.")
