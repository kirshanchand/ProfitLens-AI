
import streamlit as st
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM
)

from config import MODEL_NAME


@st.cache_resource(show_spinner="Loading Qwen AI model...")
def load_qwen_model():
    """Load Qwen once using the available hardware."""

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    dtype = (
        torch.float16
        if device == "cuda"
        else torch.float32
    )

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            dtype=dtype,
            low_cpu_mem_usage=True
        )

        model = model.to(device)
        model.eval()

        return tokenizer, model

    except Exception as error:
        raise RuntimeError(
            f"Unable to load Qwen on {device}. "
            "Check model availability, memory, "
            "and installed dependencies. "
            f"Original error: {error}"
        ) from error
