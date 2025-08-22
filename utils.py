import os
import streamlit as st
import tempfile
import re
import base64
import hashlib
from typing import Optional

# -----------------------------
# Session initialization
# -----------------------------
def init_session_state():
    """Initialize Streamlit session state variables, including quantum feature flags."""
    if "rag" not in st.session_state:
        st.session_state.rag = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "temp_dir" not in st.session_state:
        st.session_state.temp_dir = None
    if "mongo_db" not in st.session_state:
        st.session_state.mongo_db = None
    if "user" not in st.session_state:
        st.session_state.user = None
    if "auth_page" not in st.session_state:
        st.session_state.auth_page = "login"
    if "page" not in st.session_state:
        st.session_state.page = "chat"
    if "current_notebook" not in st.session_state:
        st.session_state.current_notebook = None
    if "viewing_document" not in st.session_state:
        st.session_state.viewing_document = None
    if "llm_model" not in st.session_state:
        st.session_state.llm_model = "llama3.2:latest"
    if "embedding_model" not in st.session_state:
        st.session_state.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    if "chunk_size" not in st.session_state:
        st.session_state.chunk_size = 1000
    if "chunk_overlap" not in st.session_state:
        st.session_state.chunk_overlap = 200
    if "use_gpu" not in st.session_state:
        st.session_state.use_gpu = True

    # Quantum feature flags (toggled in Settings)
    if "quantum_config" not in st.session_state:
        st.session_state.quantum_config = {
            # 1) Document ingestion & embeddings
            "enable_quantum_embeddings": False,
            # 2) Pre-LLM context selection under token budget
            "enable_quantum_context_selection": False,
            # 3) Quantum-safe storage for vectors/docs
            "enable_quantum_secure_storage": False,
            # 4) Post-LLM reranking/diversification
            "enable_quantum_rerank": False,
            # 5) System-wide quantum-inspired acceleration
            "enable_quantum_inspired_system": False,
        }

def set_quantum_flag(key: str, value: bool):
    """Set a quantum feature flag."""
    if "quantum_config" not in st.session_state:
        init_session_state()
    if key in st.session_state.quantum_config:
        st.session_state.quantum_config[key] = value

def get_quantum_flag(key: str) -> bool:
    """Get a quantum feature flag value."""
    return st.session_state.get("quantum_config", {}).get(key, False)

# -----------------------------
# File and directory helpers
# -----------------------------
def remove_directory_recursively(directory_path: str):
    """Recursively remove a directory and all its contents using os module."""
    if not os.path.exists(directory_path):
        return
    for root, dirs, files in os.walk(directory_path, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Error removing file {file_path}: {e}")
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            try:
                os.rmdir(dir_path)
            except Exception as e:
                print(f"Error removing directory {dir_path}: {e}")
    try:
        os.rmdir(directory_path)
    except Exception as e:
        print(f"Error removing top directory {directory_path}: {e}")

def cleanup_temp_files():
    """Clean up temporary files when application exits."""
    if st.session_state.get('temp_dir') and os.path.exists(st.session_state.temp_dir):
        try:
            remove_directory_recursively(st.session_state.temp_dir)
            print(f"Cleaned up temporary directory: {st.session_state.temp_dir}")
        except Exception as e:
            print(f"Error cleaning up temporary directory: {e}")

def create_temp_directory() -> str:
    """Create a temporary directory and store its path in session state."""
    temp_dir = tempfile.mkdtemp()
    st.session_state.temp_dir = temp_dir
    return temp_dir

# -----------------------------
# Password and UI helpers
# -----------------------------
def check_password_strength(password: str):
    """Check password strength and return feedback."""
    score = 0
    feedback = ""

    if len(password) < 8:
        feedback = "Password is too short. Use at least 8 characters."
        return "weak", feedback
    elif len(password) >= 12:
        score += 2
    elif len(password) >= 8:
        score += 1

    if re.search(r'[A-Z]', password) and re.search(r'[a-z]', password):
        score += 1
    else:
        feedback += "Add both uppercase and lowercase letters. "

    if re.search(r'\d', password):
        score += 1
    else:
        feedback += "Add numbers. "

    if re.search(r'[!@#$%^&*(),.?\":{}|<>]', password):
        score += 1
    else:
        feedback += "Add special characters. "

    if score >= 4:
        return "strong", "Strong password"
    elif score >= 2:
        return "medium", "Medium strength. " + feedback
    else:
        return "weak", "Weak password. " + feedback

def format_file_size(size_bytes: int) -> str:
    """Format file size from bytes to appropriate unit."""
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.1f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.1f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def get_file_icon(file_type: str) -> str:
    """Get an appropriate icon for a file type."""
    file_icons = {
        "pdf": "📕",
        "docx": "📘",
        "doc": "📘",
        "txt": "📄",
        "unknown": "📁"
    }
    return file_icons.get(file_type.lower(), "📁")

def display_error_message(error: str, suggestion: Optional[str] = None):
    """Display a styled error message with optional suggestion."""
    st.error(f"**Error:** {error}")
    if suggestion:
        st.info(f"**Suggestion:** {suggestion}")

def set_page_style():
    """Set global page styling."""
    st.markdown("""
    """, unsafe_allow_html=True)

# -----------------------------
# Quantum-safe storage (QS) stubs
# -----------------------------
# Purpose:
# - Provide encryption wrappers for at-rest vectors/documents now.
# - Keep API stable so you can swap internals with standardized PQC later (e.g., ML-KEM/ML-DSA).
# Implementation:
# - Uses AES-256-GCM with a derived key for confidentiality + integrity.
# - Stores base64 payloads conveniently in Mongo/GridFS.
# Notes:
# - Replace passphrases with KMS-managed secrets in production.
# - If pycryptodome is unavailable, functions fall back to passthrough to not break the app.

try:
    from Crypto.Cipher import AES  # pycryptodome
    from Crypto.Random import get_random_bytes
except Exception:
    AES = None
    get_random_bytes = None

def _derive_key(passphrase: str) -> bytes:
    """Derive a 32-byte key from a passphrase (replace with PBKDF2/Argon2 in production)."""
    return hashlib.sha256(passphrase.encode("utf-8")).digest()

def quantum_safe_encrypt(data: bytes, passphrase: str) -> bytes:
    """Encrypt bytes with AES-256-GCM; returns iv|tag|ciphertext. Falls back to passthrough if AES unavailable."""
    if AES is None or get_random_bytes is None:
        return data
    key = _derive_key(passphrase)
    iv = get_random_bytes(12)  # 96-bit nonce for GCM
    cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
    ciphertext, tag = cipher.encrypt_and_digest(data)
    return iv + tag + ciphertext

def quantum_safe_decrypt(payload: bytes, passphrase: str) -> Optional[bytes]:
    """Decrypt bytes produced by quantum_safe_encrypt. Returns None on failure."""
    if AES is None:
        return payload
    try:
        iv = payload[:12]
        tag = payload[12:28]
        ciphertext = payload[28:]
        key = _derive_key(passphrase)
        cipher = AES.new(key, AES.MODE_GCM, nonce=iv)
        return cipher.decrypt_and_verify(ciphertext, tag)
    except Exception:
        return None

def quantum_safe_encrypt_b64(data: bytes, passphrase: str) -> str:
    """Encrypt and return a base64 string."""
    enc = quantum_safe_encrypt(data, passphrase)
    return base64.b64encode(enc).decode("utf-8")

def quantum_safe_decrypt_b64(data_b64: str, passphrase: str) -> Optional[bytes]:
    """Decode base64 and decrypt. Returns None on failure."""
    try:
        raw = base64.b64decode(data_b64.encode("utf-8"))
        return quantum_safe_decrypt(raw, passphrase)
    except Exception:
        return None