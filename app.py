"""PocketPolly — simple chat UI."""

import base64
from pathlib import Path
import streamlit as st
from dotenv import load_dotenv

load_dotenv(override=True)

def _logo_b64() -> str:
    path = Path(__file__).parent / "langchain-color.png"
    return base64.b64encode(path.read_bytes()).decode()

st.set_page_config(
    page_title="PocketPolly",
    page_icon="🦜",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  * { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

  [data-testid="stAppViewContainer"] { background: #000000 !important; }
  [data-testid="stMainBlockContainer"] { background: #000000 !important; padding-top: 0 !important; }
  .block-container { padding-top: 0 !important; padding-bottom: 140px !important; max-width: 700px !important; }

  #MainMenu, footer, header,
  [data-testid="stDecoration"],
  [data-testid="stSidebar"],
  .stDeployButton { display: none !important; }

  /* header */
  .pg-header {
    display: flex;
    align-items: center;
    padding: 20px 0 18px;
    border-bottom: 1px solid #1c1c1c;
    margin-bottom: 32px;
  }
  .pg-brand-name { font-size: 15px; font-weight: 600; color: #ffffff; letter-spacing: -0.1px; }

  /* custom message bubbles */
  .pg-msg { display: flex; gap: 12px; margin-bottom: 16px; align-items: flex-start; }
  .pg-avatar {
    width: 32px; height: 32px; border-radius: 6px; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 13px; font-weight: 600; letter-spacing: -0.3px;
  }
  .pg-avatar-user { background: #27272a; color: #a1a1aa; }
  .pg-avatar-bot  { background: #172554; color: #93c5fd; }
  .pg-bubble {
    background: #0f0f0f;
    border: 1px solid #1c1c1c;
    border-radius: 10px;
    padding: 12px 16px;
    flex: 1;
  }
  .pg-bubble p, .pg-bubble li, .pg-bubble span, .pg-bubble div {
    color: #e4e4e7 !important;
    font-size: 14px !important;
    line-height: 1.7 !important;
    margin: 0 0 4px 0 !important;
  }
  .pg-bubble strong { color: #ffffff !important; }
  .pg-bubble ul { padding-left: 18px !important; margin: 6px 0 !important; }
  .pg-bubble li { margin-bottom: 2px !important; }

  /* entire bottom bar — white/light grey */
  [data-testid="stChatInputContainer"],
  [data-testid="stChatInputContainer"]:focus-within,
  [data-testid="stChatInputContainer"]:focus,
  [data-testid="stChatInputContainer"]:active {
    background: #f4f4f5 !important;
    border: none !important;
    border-top: 1px solid #e4e4e7 !important;
    box-shadow: none !important;
    outline: none !important;
    padding: 12px 16px !important;
  }
  [data-testid="stChatInputContainer"] > div,
  [data-testid="stChatInputContainer"] > div:focus-within {
    background: #f4f4f5 !important;
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
  }
  [data-testid="stChatInput"],
  [data-testid="stChatInput"]:focus,
  [data-testid="stChatInput"]:focus-within {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
  }
  [data-testid="stChatInput"] textarea,
  [data-testid="stChatInput"] textarea:focus,
  [data-testid="stChatInput"] textarea:active,
  [data-testid="stChatInput"] textarea:hover,
  [data-testid="stChatInput"] textarea:focus-visible {
    background: #ffffff !important;
    border: 1px solid #e4e4e7 !important;
    border-radius: 10px !important;
    color: #09090b !important;
    font-size: 14px !important;
    line-height: 1.5 !important;
    padding: 10px 14px !important;
    min-height: 44px !important;
    caret-color: #09090b !important;
    box-shadow: none !important;
    outline: none !important;
    -webkit-box-shadow: none !important;
  }
  [data-testid="stChatInput"] textarea::placeholder { color: #a1a1aa !important; }

  /* suggestion buttons */
  .stButton button {
    background: #0f0f0f !important;
    border: 1px solid #1c1c1c !important;
    color: #71717a !important;
    border-radius: 8px !important;
    font-size: 13px !important;
    padding: 10px 14px !important;
    width: 100% !important;
    text-align: left !important;
    transition: border-color 0.12s, color 0.12s !important;
  }
  .stButton button:hover {
    border-color: #3b82f6 !important;
    color: #a1a1aa !important;
  }

  .stSpinner > div { border-top-color: #3b82f6 !important; }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="pg-header">
  <div style="display:flex; align-items:center; gap:9px;">
    <img src="data:image/png;base64,{_logo_b64()}" width="26" height="26" style="display:block;"/>
    <span class="pg-brand-name">LangChain</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Empty state ───────────────────────────────────────────────────────────
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align:center; padding: 48px 0 36px;">
      <svg width="72" height="72" viewBox="0 0 72 72" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M26 54 Q16 64 13 72" stroke="#5ab4d6" stroke-width="3" stroke-linecap="round" fill="none"/>
        <path d="M30 56 Q22 66 20 72" stroke="#7ecce8" stroke-width="2" stroke-linecap="round" fill="none"/>
        <ellipse cx="34" cy="40" rx="13" ry="16" fill="#5ab4d6"/>
        <ellipse cx="22" cy="42" rx="7" ry="12" fill="#7ecce8" transform="rotate(-12 22 42)"/>
        <ellipse cx="36" cy="44" rx="7" ry="10" fill="#a8dff0"/>
        <circle cx="36" cy="22" r="12" fill="#7ecce8"/>
        <circle cx="40" cy="19" r="4" fill="#fafafa"/>
        <circle cx="41" cy="19" r="2" fill="#09090b"/>
        <circle cx="40.5" cy="18.5" r="0.6" fill="#fafafa"/>
        <path d="M44 23 Q50 24 44 27 Z" fill="#a8dff0"/>
        <path d="M44 25 Q49 26 44 27 Z" fill="#5ab4d6"/>
        <line x1="30" y1="56" x2="27" y2="63" stroke="#5ab4d6" stroke-width="2" stroke-linecap="round"/>
        <line x1="35" y1="57" x2="35" y2="64" stroke="#5ab4d6" stroke-width="2" stroke-linecap="round"/>
        <line x1="27" y1="63" x2="24" y2="66" stroke="#5ab4d6" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="27" y1="63" x2="28" y2="66" stroke="#5ab4d6" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="35" y1="64" x2="32" y2="67" stroke="#5ab4d6" stroke-width="1.5" stroke-linecap="round"/>
        <line x1="35" y1="64" x2="37" y2="67" stroke="#5ab4d6" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
      <div style="font-size:22px; font-weight:700; color:#ffffff; margin-top:16px; letter-spacing:-0.3px;">PocketPolly</div>
      <div style="font-size:13px; color:#52525b; margin-top:6px;">Parrot expert agent built on LangGraph</div>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        "Can parrots eat avocado?",
        "How often should I bathe my parrot?",
        "How long do African Grey parrots live?",
        "Is chocolate safe for my parrot?",
    ]
    cols = st.columns(2)
    for i, text in enumerate(suggestions):
        if cols[i % 2].button(text, key=f"sug_{i}"):
            st.session_state.pending = text
            st.rerun()

# ── Render messages as custom HTML ────────────────────────────────────────
import markdown as md_lib

for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"""
        <div class="pg-msg">
          <div class="pg-avatar pg-avatar-user">You</div>
          <div class="pg-bubble"><p>{msg["content"]}</p></div>
        </div>
        """, unsafe_allow_html=True)
    else:
        body = md_lib.markdown(msg["content"])
        st.markdown(f"""
        <div class="pg-msg">
          <div class="pg-avatar pg-avatar-bot">PG</div>
          <div class="pg-bubble">{body}</div>
        </div>
        """, unsafe_allow_html=True)

# ── Input ─────────────────────────────────────────────────────────────────
user_input = st.chat_input("Message PocketPolly...")

if "pending" in st.session_state:
    user_input = st.session_state.pop("pending")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Show user bubble immediately
    st.markdown(f"""
    <div class="pg-msg">
      <div class="pg-avatar pg-avatar-user">You</div>
      <div class="pg-bubble"><p>{user_input}</p></div>
    </div>
    """, unsafe_allow_html=True)

    # Stream assistant response into a placeholder
    st.markdown("""
    <div class="pg-msg" id="streaming-msg">
      <div class="pg-avatar pg-avatar-bot">PG</div>
      <div class="pg-bubble" id="streaming-bubble">
    """, unsafe_allow_html=True)

    placeholder = st.empty()
    response = ""

    try:
        from agent.agent import stream_agent
        for chunk in stream_agent(question=user_input):
            response += chunk
            placeholder.markdown(
                f'<div style="color:#e4e4e7;font-size:14px;line-height:1.7;">{response}</div>',
                unsafe_allow_html=True,
            )
    except Exception as e:
        response = f"Error: {e}"
        placeholder.markdown(response)

    st.markdown("</div></div>", unsafe_allow_html=True)

    st.session_state.messages.append({"role": "assistant", "content": response})
    st.rerun()
