import base64
import io
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "0")

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from huggingface_hub import snapshot_download

st.set_page_config(
    page_title="Cortexaa | Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(BASE_DIR / "best_brisc_cnn_efficientnet.keras"),
    )
)

DATASET_PATH = Path(os.environ.get("BRISC_PATH", str(BASE_DIR / "brisc2025")))
TRAIN_PATH = DATASET_PATH / "classification_task" / "train"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
FALLBACK_CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]
HF_DATASET_REPO = "Gurukant12345/brisc2025-test"


def find_class_root(root):
    root = Path(root)
    expected = set(FALLBACK_CLASSES)
    candidates = [root] + [p for p in root.rglob("*") if p.is_dir()]
    for folder in candidates:
        if folder.name.startswith("."):
            continue
        subdirs = {p.name for p in folder.iterdir() if p.is_dir()}
        if len(expected & subdirs) >= 2:
            return folder
    return root


@st.cache_resource(show_spinner="Downloading test dataset from Hugging Face...")
def download_test_dataset():
    token = None
    try:
        token = st.secrets.get("HF_TOKEN", None)
    except Exception:
        token = os.environ.get("HF_TOKEN")
    local_dir = snapshot_download(
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=token,
    )
    return str(find_class_root(local_dir))


def get_test_path():
    try:
        return Path(download_test_dataset()), None
    except Exception as exc:
        return Path("__missing__"), f"{type(exc).__name__}: {exc}"


def html(markup: str):
    cleaned = "\n".join(
        line.strip() for line in markup.splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


html(
    """
<style>
@import url('
