# PROJECT TITLE - AI Text Generator

# Install Dependencies
!pip install transformers torch streamlit pyngrok==4.1.1

# Import Packages & Download Models
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM

# Load Sentiment Analysis Model
sentiment_pipeline = pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english"
)
# Load Text Generation Model
generator_name = "gpt2"
generator_tokenizer = AutoTokenizer.from_pretrained(generator_name)
generator_model = AutoModelForCausalLM.from_pretrained(generator_name)

# Streamlit UI & Backend Logic
app_code = '''
import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import re

# ================================
# Load Models
# ================================
@st.cache_resource(show_spinner=False)
def load_models():
    # Sentiment Classifier (DistilBERT)
    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english"
    )

    # GPT-2 Text Generator
    generator_name = "gpt2"
    tokenizer = AutoTokenizer.from_pretrained(generator_name)
    model = AutoModelForCausalLM.from_pretrained(generator_name)

    return sentiment_pipeline, tokenizer, model

sentiment_pipeline, tokenizer, model = load_models()

# ================================
# Helper Functions
# ================================
def clean_prompt(text):
    """
    Cleans the user prompt by removing instructions or meta-comments.
    """
    lines = text.split("\\n")
    content_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Remove lines with instruction keywords
        if re.search(r"\\b(write|be positive|if you|try|exercise|too long|not overly|uplifting|motivational)\\b", line, re.IGNORECASE):
            continue
        content_lines.append(line)
    return " ".join(content_lines)

def detect_sentiment(text):
    """
    Detects sentiment using DistilBERT.
    Returns 'positive', 'negative', or 'neutral'.
    """
    result = sentiment_pipeline(text)[0]
    label = result["label"].lower()
    if label == "positive":
        return "positive"
    elif label == "negative":
        return "negative"
    else:
        return "neutral"

def generate_text(prompt, sentiment, max_length=200):
    """
    Generates text aligned with the detected sentiment using GPT-2.
    """
    instructions = {
        "positive": "Write a positive, uplifting paragraph about: ",
        "negative": "Write a negative, sad paragraph about: ",
        "neutral": "Write a neutral, factual paragraph about: "
    }
    final_prompt = instructions.get(sentiment, instructions["neutral"]) + prompt
    input_ids = tokenizer.encode(final_prompt, return_tensors="pt")
    output = model.generate(
        input_ids,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        no_repeat_ngram_size=3,
        do_sample=True
    )
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ================================
# Streamlit UI
# ================================
st.set_page_config(
    page_title="AI Sentiment-Based Text Generator",
    layout="wide",
    page_icon="🤖"
)

# Custom CSS
st.markdown("""
<style>
.title { color: #4B0082; font-size: 36px; font-weight: bold; text-align: center; }
.subtitle { color: #6A5ACD; font-size: 18px; text-align: center; }
.sentiment { font-weight: bold; color: white; padding: 5px 10px; border-radius: 5px; display: inline-block; }
.positive { background-color: #28a745; }
.negative { background-color: #dc3545; }
.neutral { background-color: #ffc107; color: black; }
.generated-text { background-color: #f0f2f6; color: #000000; padding: 15px; border-radius: 10px; font-size: 16px; line-height: 1.6; white-space: pre-wrap; }
.instructions { color: #444; font-size: 14px; margin-bottom: 10px; }
</style>
""", unsafe_allow_html=True)

# Page Header
st.markdown('<div class="title">🎯 AI Sentiment-Based Text Generator</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Enter a prompt and generate sentiment-aligned text instantly</div>', unsafe_allow_html=True)
st.markdown("---")

# User Input
st.subheader("📝 Enter Your Prompt")
st.markdown('<div class="instructions">Instructions or extra guidance will be automatically removed for cleaner output.</div>', unsafe_allow_html=True)
user_prompt = st.text_area("", height=150, placeholder="Type your prompt here...")

st.subheader("🎭 Sentiment Selection (Optional)")
sentiment_choice = st.selectbox("Choose sentiment (or Auto Detect):", ["Auto Detect", "positive", "negative", "neutral"])

st.subheader("📏 Generated Text Length")
max_len = st.slider("Select length (in tokens):", min_value=50, max_value=500, value=200, step=10)

# Generate Button
if st.button("🚀 Generate Text"):
    if not user_prompt.strip():
        st.warning("⚠️ Please enter a prompt before generating text.")
    else:
        with st.spinner("Cleaning prompt, detecting sentiment, and generating text..."):
            cleaned_prompt = clean_prompt(user_prompt)
            sentiment = detect_sentiment(cleaned_prompt) if sentiment_choice=="Auto Detect" else sentiment_choice
            generated = generate_text(cleaned_prompt, sentiment, max_length=max_len)

        st.markdown("### 🧠 Detected Sentiment")
        st.markdown(f'<span class="sentiment {sentiment}">{sentiment.upper()}</span>', unsafe_allow_html=True)
        st.markdown("### 📝 Generated Text")
        st.markdown(f'<div class="generated-text">{generated}</div>', unsafe_allow_html=True)
'''
# Write to app.py
with open("app.py", "w") as f:
    f.write(app_code)
print("✅ app.py created successfully!")

# Streamlit App Deployment

# Install Streamlit + Ngrok v3
!pip install -q streamlit groq python-docx pypdf

# Download ngrok v3
!wget -q -O ngrok.zip https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.zip
!unzip -qo ngrok.zip
!chmod +x ngrok
!mv ngrok /usr/local/bin/ngrok

# Configure Environment Variables
import os, time, subprocess, requests

# Groq API key
os.environ["GROQ_API_KEY"] = "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"

# Ngrok auth token
NGROK_AUTH_TOKEN = "2z0Oqv0tD166fELGCHwV2gLZwq1_2G2zUQRSs6C27k9vdzxwq"
!ngrok config add-authtoken $NGROK_AUTH_TOKEN

# Create Log Directory
LOG_DIR = "/content/logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Kill previous sessions
subprocess.run(["pkill", "-f", "streamlit"], stderr=subprocess.PIPE)
subprocess.run(["pkill", "-f", "ngrok"], stderr=subprocess.PIPE)

# Start Streamlit App
APP_FILE = "app.py"
!streamlit run $APP_FILE --server.port 8501 --server.address 0.0.0.0 > {LOG_DIR}/app_log.txt 2>&1 &
print("🔄 Starting Streamlit... please wait.")
time.sleep(7)

# Start Ngrok Tunnel
print("🔄 Starting ngrok tunnel...")
ngrok_process = subprocess.Popen(["ngrok", "http", "8501"])
time.sleep(5)

# Fetch public URL
try:
    tunnel_info = requests.get("http://localhost:4040/api/tunnels").json()
    public_url = tunnel_info["tunnels"][0]["public_url"]
    print("🚀 Your Groq-powered Streamlit App is LIVE at:", public_url)
except:
    print("❌ Could not retrieve ngrok URL. Check logs below.")
