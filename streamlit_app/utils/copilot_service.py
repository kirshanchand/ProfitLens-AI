
import importlib
import threading
from pathlib import Path

from utils import copilot_core
from utils.qwen_loader import load_qwen_model

_copilot_lock = threading.RLock()
_loaded_version = None


def ask_profitlens(question):
    """Answer using the latest verified Copilot logic."""

    global _loaded_version

    with _copilot_lock:

        # Reload Copilot logic only when its source changes.
        module_path = Path(copilot_core.__file__)
        current_version = module_path.stat().st_mtime_ns

        if current_version != _loaded_version:
            importlib.reload(copilot_core)
            _loaded_version = current_version

        # Load or reuse the cached Qwen model.
        tokenizer, model = load_qwen_model()

        copilot_core.configure_qwen(
            tokenizer,
            model
        )

        # Reconnect business data if the module was reloaded.
        if "conn_ai" not in vars(copilot_core):
            copilot_core.initialize_business_data()

        return copilot_core.answer_business_question(
            question
        )
