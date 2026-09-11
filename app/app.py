"""Local Streamlit interface with cached model and serialized GPU access."""
import sys
import threading
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import streamlit as st
import torch
from src.inference import load_tutor, answer
from src.utils import config

st.set_page_config(page_title="Arabic AI Tutor", page_icon="📚")
st.markdown('<style>[data-testid="stChatMessage"] {direction:rtl;text-align:right;} '
            'pre,code {direction:ltr;text-align:left;}</style>', unsafe_allow_html=True)
st.title("Arabic AI Tutor")
cfg = config()
with st.sidebar:
    st.write("Model:", cfg["model_name"])
    adapter = st.text_input("Adapter path", cfg["output_dir"])
    temperature = st.slider("Temperature", 0.0, 1.5, 0.7, 0.1)
    max_tokens = st.slider("Max tokens", 64, 768, 384, 64)
    if st.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()


@st.cache_resource(max_entries=1)
def resource(adapter_path):
    model, tokenizer = load_tutor(cfg, adapter_path)
    return model, tokenizer, threading.Lock()


if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
if question := st.chat_input("اسأل عن الذكاء الاصطناعي"):
    with st.chat_message("user"):
        st.markdown(question)
    try:
        with st.spinner("جارٍ إعداد الشرح..."):
            model, tokenizer, lock = resource(adapter)
            with lock:
                response = answer(model, tokenizer, question, history=st.session_state.messages,
                                  temperature=temperature, max_new_tokens=max_tokens)
        st.session_state.messages.extend([{"role": "user", "content": question},
                                           {"role": "assistant", "content": response}])
        with st.chat_message("assistant"):
            st.markdown(response)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        st.error("ذاكرة GPU غير كافية. قلّل عدد الرموز وأغلق التطبيقات التي تستخدم GPU.")
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(str(exc))
