import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
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

import os

os.environ["HF_HUB_DISABLE_XET"] = "1"

from huggingface_hub import snapshot_download

# =========================================================
# PAGE CONFIGURATION
# =========================================================
st.set_page_config(
    page_title="Cortexa | Brain MRI Analysis",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent

# Fine-tuned EfficientNetB0 checkpoint downloaded from Colab.
MODEL_PATH = Path(
    os.environ.get(
        "MODEL_PATH",
        str(BASE_DIR / "best_brisc_cnn_efficientnet.keras"),
    )
)

DATASET_PATH = Path(os.environ.get("BRISC_PATH", str(BASE_DIR / "brisc2025")))
TRAIN_PATH = DATASET_PATH / "classification_task" / "train"


@st.cache_resource
def download_test_dataset():
    return snapshot_download(
        repo_id="Gurukant12345/brisc2025-test",
        repo_type="dataset",
        max_workers=4
    )

TEST_PATH = Path(download_test_dataset())

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
FALLBACK_CLASSES = ["glioma", "meningioma", "no_tumor", "pituitary"]


# =========================================================
# HTML RENDER HELPER  (THE FIX)
# =========================================================
def html(markup: str):
    """
    Render raw HTML safely in Streamlit.

    Streamlit's Markdown parser treats any line indented by 4+ spaces (or
    separated by blank lines) as a code block, which made the cards show raw
    <div> text. Stripping indentation and blank lines avoids that.
    """
    cleaned = "\n".join(
        line.strip() for line in markup.splitlines() if line.strip()
    )
    st.markdown(cleaned, unsafe_allow_html=True)


# =========================================================
# RESPONSIVE DARK THEME
# =========================================================
html(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root {
    --bg: #080d18;
    --panel: #111d31;
    --panel-2: #15243b;
    --border: rgba(148, 163, 184, 0.18);
    --text: #e8f0ff;
    --muted: #9aacc8;
    --cyan: #70e1f5;
    --purple: #a78bfa;
}

html, body, [data-testid="stAppViewContainer"], .stApp {
    background: #080d18 !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse at 8% 0%, rgba(37,99,235,.17), transparent 35%),
        radial-gradient(ellipse at 95% 12%, rgba(124,58,237,.14), transparent 32%),
        linear-gradient(140deg, #080d18 0%, #0c1424 55%, #080d18 100%) !important;
}

header[data-testid="stHeader"] {
    background: rgba(8, 13, 24, 0.96) !important;
    border-bottom: 1px solid rgba(148,163,184,.10);
}
[data-testid="stToolbar"] { background: transparent !important; }
[data-testid="stDecoration"] { background: transparent !important; }

.main .block-container {
    max-width: 1420px !important;
    padding: 2rem 2.4rem 3rem !important;
}

h1, h2, h3, h4 {
    color: #f2f6ff !important;
    font-family: 'Space Grotesk', sans-serif !important;
    letter-spacing: -0.45px;
    overflow-wrap: anywhere;
}

p, li, label, .stMarkdown, [data-testid="stCaptionContainer"] { color: var(--text); }

section[data-testid="stSidebar"] {
    background: #0b1424 !important;
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] > div { background: #0b1424 !important; }

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] label p,
section[data-testid="stSidebar"] span {
    color: #e8f0ff !important;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: var(--muted) !important; }

section[data-testid="stSidebar"] [role="radiogroup"] label {
    background: transparent;
    border-radius: 9px;
    padding: 7px 9px;
}
section[data-testid="stSidebar"] [role="radiogroup"] label:hover { background: rgba(112,225,245,.08); }

.brand {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 23px;
    font-weight: 700;
    color: #f2f6ff;
    margin: 2px 0 4px;
}
.muted { color: var(--muted) !important; font-size: 13px; line-height: 1.65; }

.hero {
    padding: 30px 32px;
    border: 1px solid rgba(112,225,245,.23);
    border-radius: 22px;
    background:
        radial-gradient(circle at 88% 8%, rgba(112,225,245,.12), transparent 32%),
        linear-gradient(120deg, rgba(20,38,65,.98), rgba(17,24,45,.94));
    box-shadow: 0 15px 45px rgba(0,0,0,.17);
    margin-bottom: 26px;
}
.eyebrow {
    color: var(--cyan) !important;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 1.8px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.hero-title {
    color: #f5f8ff !important;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(30px, 4vw, 46px);
    font-weight: 700;
    line-height: 1.12;
    margin: 8px 0 12px;
}
.hero-sub {
    color: #b5c4dc !important;
    max-width: 780px;
    font-size: 15px;
    line-height: 1.7;
}
.section-title {
    color: #f2f6ff !important;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 22px;
    font-weight: 650;
    margin: 12px 0 7px;
}
.section-subtitle {
    color: var(--muted) !important;
    font-size: 13px;
    line-height: 1.6;
    margin-bottom: 16px;
}
.panel {
    box-sizing: border-box;
    height: 100%;
    padding: 21px;
    border: 1px solid var(--border);
    border-radius: 18px;
    background: linear-gradient(145deg, rgba(22,36,59,.96), rgba(14,23,40,.96));
    box-shadow: 0 10px 28px rgba(0,0,0,.13);
    margin-bottom: 14px;
    overflow-wrap: anywhere;
}
.panel h2, .panel h3 { margin: 4px 0 8px; }
.category-card {
    min-height: 150px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}
.category-name {
    color: #f5f8ff !important;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(19px, 1.8vw, 25px);
    font-weight: 700;
    line-height: 1.2;
    overflow-wrap: normal;
    word-break: normal;
}
.category-tag { color: var(--cyan) !important; font-size: 12px; margin-top: 12px; }

.metric-card {
    box-sizing: border-box;
    height: 100%;
    min-height: 126px;
    padding: 19px;
    border: 1px solid var(--border);
    border-radius: 17px;
    background: linear-gradient(145deg, rgba(25,39,64,.96), rgba(14,23,40,.96));
    margin-bottom: 14px;
}
.metric-label { color: #a9bad4 !important; font-size: 12px; margin-bottom: 9px; }
.metric-value {
    color: #f4f7ff !important;
    font-family: 'Space Grotesk', sans-serif;
    font-size: clamp(22px, 2.3vw, 30px);
    font-weight: 700;
    line-height: 1.2;
    overflow-wrap: anywhere;
}
.metric-caption { color: var(--cyan) !important; font-size: 11px; margin-top: 7px; line-height: 1.45; }

.stButton > button {
    background: linear-gradient(100deg, #67e8f9, #a5b4fc) !important;
    color: #08111f !important;
    border: 0 !important;
    border-radius: 11px !important;
    font-weight: 700 !important;
    min-height: 44px;
}
.stButton > button:hover { box-shadow: 0 0 22px rgba(103,232,249,.2); transform: translateY(-1px); }
.stDownloadButton > button { border-radius: 10px !important; }

[data-testid="stFileUploader"] {
    border: 1px dashed rgba(112,225,245,.38);
    border-radius: 14px;
    padding: 10px;
    background: rgba(16,29,49,.58);
}
[data-testid="stFileUploader"] * { color: #e8f0ff !important; }

[data-testid="stAlert"] {
    background: rgba(20,38,65,.96) !important;
    color: #e8f0ff !important;
    border: 1px solid rgba(112,225,245,.17) !important;
    border-radius: 13px !important;
}
[data-testid="stAlert"] p { color: #e8f0ff !important; }

[data-testid="stDataFrame"], [data-testid="stTable"] {
    border: 1px solid var(--border);
    border-radius: 12px;
    overflow: hidden;
}
[data-baseweb="select"] > div, [data-baseweb="input"] > div { background: #132139 !important; }
hr { border-color: rgba(148,163,184,.15) !important; }
footer, #MainMenu { visibility: hidden; }

@media (max-width: 900px) {
    .main .block-container { padding: 1.4rem 1.2rem 2rem !important; }
    .hero { padding: 23px; }
    .category-card { min-height: 135px; }
}
@media (max-width: 600px) {
    .main .block-container { padding: 1rem .8rem 1.5rem !important; }
    .hero { padding: 19px; border-radius: 16px; }
    .section-title { font-size: 20px; }
}
</style>
"""
)


# =========================================================
# HELPERS
# =========================================================
def section_heading(title, subtitle=None):
    html(f'<div class="section-title">{title}</div>')
    if subtitle:
        html(f'<div class="section-subtitle">{subtitle}</div>')


def hero(title, subtitle, eyebrow="Cortexa"):
    html(
        f"""
<div class="hero">
<div class="eyebrow">{eyebrow}</div>
<div class="hero-title">{title}</div>
<div class="hero-sub">{subtitle}</div>
</div>
"""
    )


def metric_card(label, value, caption=""):
    html(
        f"""
<div class="metric-card">
<div class="metric-label">{label}</div>
<div class="metric-value">{value}</div>
<div class="metric-caption">{caption}</div>
</div>
"""
    )


def get_class_names():
    if TRAIN_PATH.exists():
        names = sorted(p.name for p in TRAIN_PATH.iterdir() if p.is_dir())
        if names:
            return names
    return FALLBACK_CLASSES


# =========================================================
# MODEL LOADING AND TEST EVALUATION
# =========================================================
@st.cache_resource
def load_model(model_path, modified_time):
    del modified_time
    return tf.keras.models.load_model(model_path)


@st.cache_data(show_spinner=False)
def evaluate_test_set(model_path, model_mtime, test_path):
    del model_mtime

    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_path,
        image_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        shuffle=False,
        label_mode="int",
    )
    class_names = test_ds.class_names
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])

    eval_model = tf.keras.models.load_model(model_path)
    probabilities = eval_model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)

    if probabilities.shape[1] != len(class_names):
        raise ValueError(
            "The model output count does not match the number of test classes. "
            f"Model outputs: {probabilities.shape[1]}, "
            f"dataset classes: {len(class_names)}."
        )

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "class_names": class_names,
        "y_true": y_true,
        "y_pred": y_pred,
        "cm": cm,
        "report": report,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="macro", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="macro", zero_division=0),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "count": len(y_true),
    }


# =========================================================
# SIDEBAR
# =========================================================
class_names = get_class_names()

with st.sidebar:
    html('<div class="brand">🧠 Cortexa</div>')
    html('<div class="muted">AI-assisted brain MRI research interface</div>')
    st.divider()

    page = st.radio(
        "NAVIGATION",
        ["Overview", "MRI Prediction", "Model Dashboard", "About Model"],
        label_visibility="visible",
    )

    st.divider()
    st.markdown("**MODEL STATUS**")
    if MODEL_PATH.exists():
        st.success("Fine-tuned EfficientNetB0 model found")
    else:
        st.error("EfficientNetB0 model file missing")

    html('<div class="muted">BRISC 2025 · Four-class classification</div>')


# =========================================================
# MODEL CHECK
# =========================================================
if not MODEL_PATH.exists():
    st.title("🧠 Cortexa")
    st.error(
        f"Model not found at:\n\n{MODEL_PATH}\n\n"
        "Place best_brisc_cnn_efficientnet.keras beside app.py, "
        "or set the MODEL_PATH environment variable."
    )
    st.stop()

try:
    model = load_model(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
except Exception as exc:
    st.error(f"Could not load the model: {exc}")
    st.stop()


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================
if page == "Overview":
    hero(
        "See beyond the scan.",
        "Explore brain MRI image classification using a fine-tuned "
        "EfficientNetB0 model, with prediction insights and test-set analytics.",
        "BRAIN MRI INTELLIGENCE PLATFORM",
    )

    section_heading("System overview", "A simple workflow from image upload to model output.")

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        html(
            """
<div class="panel">
<div class="eyebrow">01 / INPUT</div>
<h3>Upload MRI</h3>
<div class="muted">Choose a brain MRI image in JPG, JPEG or PNG format.</div>
</div>
"""
        )
    with c2:
        html(
            """
<div class="panel">
<div class="eyebrow">02 / ANALYSIS</div>
<h3>Feature extraction</h3>
<div class="muted">The pretrained EfficientNetB0 backbone extracts image features for classification.</div>
</div>
"""
        )
    with c3:
        html(
            """
<div class="panel">
<div class="eyebrow">03 / OUTPUT</div>
<h3>Prediction insights</h3>
<div class="muted">View the predicted class and output scores for all four categories.</div>
</div>
"""
        )

    st.markdown("---")
    section_heading("Classification categories")

    category_columns = st.columns(4, gap="medium")
    category_descriptions = {
        "glioma": "BRISC 2025 category",
        "meningioma": "BRISC 2025 category",
        "no_tumor": "BRISC 2025 category",
        "pituitary": "BRISC 2025 category",
    }

    for col, name in zip(category_columns, class_names):
        display_name = name.replace("_", " ").title()
        with col:
            html(
                f"""
<div class="panel category-card">
<div class="eyebrow">MRI CLASS</div>
<div class="category-name">{display_name}</div>
<div class="category-tag">{category_descriptions.get(name, "BRISC 2025 category")}</div>
</div>
"""
            )

    st.info(
        "Research use only. This model is not a medical diagnostic device "
        "and must not be used to make clinical decisions."
    )


# =========================================================
# PAGE 2: MRI PREDICTION
# =========================================================
elif page == "MRI Prediction":
    hero(
        "MRI Analysis Lab",
        "Upload a brain MRI image to run inference with the fine-tuned EfficientNetB0 model.",
        "LIVE MODEL INFERENCE",
    )

    upload_col, result_col = st.columns([1, 1.1], gap="large")

    with upload_col:
        section_heading("01. Upload image", "Supported formats: JPG, JPEG and PNG.")

        uploaded_file = st.file_uploader(
            "Choose a brain MRI image",
            type=["jpg", "jpeg", "png"],
            help="Upload an image for model inference.",
        )

        image = None
        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
                
                # Fix: Convert Pillow image to bytes to render correctly in st.image
                import io
                buf = io.BytesIO()
                image.save(buf, format="JPEG")
                byte_im = buf.getvalue()

                st.image(byte_im, caption="Uploaded MRI image")
                st.caption(f"Original dimensions: {image.width} × {image.height}")
            except Exception as exc:
                st.error(f"Could not read this image: {exc}")

        if image is None:
            html(
                """
<div class="panel">
<div class="eyebrow">AWAITING INPUT</div>
<h3>Your scan appears here</h3>
<div class="muted">Upload an MRI image using the control above to begin.</div>
</div>
"""
            )

    with result_col:
        section_heading("02. Prediction results")

        if image is not None and st.button("✦ Analyze MRI image", use_container_width=True):
            with st.spinner("Processing image through EfficientNetB0..."):
                try:
                    resized = image.resize(IMG_SIZE, Image.Resampling.BILINEAR)
                    image_array = np.asarray(resized, dtype=np.float32)
                    image_array = np.expand_dims(image_array, axis=0)

                    # Do not divide by 255 here. The Keras EfficientNetB0
                    # model includes its own input rescaling.
                    probabilities = model.predict(image_array, verbose=0)[0]

                    if len(probabilities) != len(class_names):
                        raise ValueError("Model output count does not match the class list.")

                    predicted_index = int(np.argmax(probabilities))
                    st.session_state["last_prediction"] = {
                        "filename": uploaded_file.name,
                        "predicted_class": class_names[predicted_index],
                        "confidence": float(probabilities[predicted_index]),
                        "probabilities": probabilities.tolist(),
                    }
                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")

        result = st.session_state.get("last_prediction")

        if result:
            pretty_name = result["predicted_class"].replace("_", " ").title()
            html(
                f"""
<div class="panel">
<div class="eyebrow">MODEL OUTPUT</div>
<h2>{pretty_name}</h2>
<div class="muted">Highest-scoring class from the model's four outputs.</div>
<div class="metric-value">{result['confidence'] * 100:.2f}%</div>
<div class="metric-caption">Model confidence score — not clinical certainty</div>
</div>
"""
            )

            st.markdown("**Class output scores**")
            score_df = pd.DataFrame(
                {
                    "Class": [n.replace("_", " ").title() for n in class_names],
                    "Probability": result["probabilities"],
                }
            ).sort_values("Probability", ascending=True)

            fig = px.bar(
                score_df,
                x="Probability",
                y="Class",
                orientation="h",
                text=score_df["Probability"].map(lambda v: f"{v * 100:.1f}%"),
                color="Probability",
                color_continuous_scale=["#334155", "#38bdf8", "#a78bfa"],
                range_x=[0, 1],
            )
            fig.update_layout(
                height=310,
                template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                showlegend=False,
                coloraxis_showscale=False,
                margin=dict(l=10, r=20, t=15, b=10),
                xaxis_title="Model output score",
                yaxis_title="",
            )
            fig.update_traces(textposition="outside", cliponaxis=False)
            st.plotly_chart(fig, use_container_width=True)

            st.caption(f"Image analyzed: {result['filename']}")
            st.warning(
                "This is an experimental classifier. A predicted class or "
                "high confidence score does not confirm or exclude a tumor. "
                "Consult a qualified radiologist or clinician."
            )
        else:
            html(
                """
<div class="panel">
<div class="eyebrow">RESULTS PANEL</div>
<h3>Ready for inference</h3>
<div class="muted">Upload an MRI and select Analyze MRI image to view the predicted class and all four output scores.</div>
</div>
"""
            )


# =========================================================
# PAGE 3: MODEL DASHBOARD
# =========================================================
elif page == "Model Dashboard":
    hero(
        "Model performance",
        "Evaluate the fine-tuned EfficientNetB0 model on the configured BRISC 2025 test set.",
        "EVALUATION & ANALYTICS",
    )

    if not TEST_PATH.exists():
        st.warning(
            f"Test directory not found: {TEST_PATH}\n\n"
            "Place the BRISC 2025 classification test folder in the configured dataset directory."
        )
        st.stop()

    try:
        with st.spinner("Evaluating test images. This may take a moment..."):
            metrics = evaluate_test_set(
                str(MODEL_PATH),
                MODEL_PATH.stat().st_mtime,
                str(TEST_PATH),
            )
    except Exception as exc:
        st.error(f"Could not evaluate the test dataset: {exc}")
        st.stop()

    if metrics is None:
        st.warning("Test dataset is unavailable.")
        st.stop()

    k1, k2, k3, k4 = st.columns(4, gap="medium")
    with k1:
        metric_card("Test accuracy", f"{metrics['accuracy'] * 100:.2f}%",
                    "Correct predictions / all test images")
    with k2:
        metric_card("Macro precision", f"{metrics['precision'] * 100:.2f}%",
                    "Average across classes")
    with k3:
        metric_card("Macro recall", f"{metrics['recall'] * 100:.2f}%",
                    "Average across classes")
    with k4:
        metric_card("Macro F1-score", f"{metrics['f1'] * 100:.2f}%",
                    "Precision-recall balance")

    st.markdown("---")
    section_heading(
        "Confusion matrix",
        "Rows represent actual classes; columns represent predicted classes.",
    )

    names = [n.replace("_", " ").title() for n in metrics["class_names"]]

    cm_fig = go.Figure(
        data=go.Heatmap(
            z=metrics["cm"],
            x=names,
            y=names,
            colorscale=[
                [0.0, "#101827"],
                [0.35, "#164e63"],
                [0.7, "#0e7490"],
                [1.0, "#67e8f9"],
            ],
            text=metrics["cm"],
            texttemplate="%{text}",
            textfont={"size": 16},
            hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Images: %{z}<extra></extra>",
            colorbar=dict(title="Images"),
        )
    )
    cm_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=440,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Predicted class",
        yaxis_title="Actual class",
    )
    st.plotly_chart(cm_fig, use_container_width=True)

    st.markdown("---")
    section_heading("Class-wise performance")

    report = metrics["report"]
    rows = []
    for name in metrics["class_names"]:
        rows.append(
            {
                "Class": name.replace("_", " ").title(),
                "Precision": report[name]["precision"],
                "Recall": report[name]["recall"],
                "F1-score": report[name]["f1-score"],
                "Test images": int(report[name]["support"]),
            }
        )

    class_df = pd.DataFrame(rows)
    display_df = class_df.copy()
    for column in ["Precision", "Recall", "F1-score"]:
        display_df[column] = display_df[column].map(lambda v: f"{v:.3f}")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    bar_df = class_df.melt(
        id_vars="Class",
        value_vars=["Precision", "Recall", "F1-score"],
        var_name="Metric",
        value_name="Score",
    )
    bar_fig = px.bar(
        bar_df,
        x="Class",
        y="Score",
        color="Metric",
        barmode="group",
        color_discrete_sequence=["#67e8f9", "#a78bfa", "#34d399"],
        range_y=[0, 1],
        text_auto=".2f",
    )
    bar_fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        margin=dict(l=10, r=10, t=20, b=20),
        legend_title="",
        yaxis_title="Score",
        xaxis_title="",
    )
    st.plotly_chart(bar_fig, use_container_width=True)

    st.download_button(
        "Download class-wise metrics CSV",
        data=class_df.to_csv(index=False).encode("utf-8"),
        file_name="brain_tumor_classification_metrics.csv",
        mime="text/csv",
    )

    st.caption(
        f"Evaluation set: {metrics['count']} images. "
        "Metrics are recomputed using the model and test folder currently configured."
    )
    st.info(
        "The earlier custom CNN achieved 83.6% test accuracy, while the "
        "fine-tuned EfficientNetB0 achieved approximately 84.0% in the "
        "notebook evaluation. This dashboard independently recalculates "
        "the score using the local model and test directory; results can "
        "differ if the checkpoint, preprocessing, or dataset differs."
    )


# =========================================================
# PAGE 4: ABOUT MODEL
# =========================================================
elif page == "About Model":
    hero(
        "Inside EfficientNetB0",
        "A concise overview of the transfer-learning architecture used in this project.",
        "MODEL ARCHITECTURE",
    )

    section_heading("Architecture pipeline")

    architecture = [
        ("01", "Input image", "224 × 224 × 3 RGB image"),
        ("02", "Data augmentation", "Random rotation, zoom, translation and contrast during training"),
        ("03", "EfficientNetB0 backbone", "ImageNet-pretrained feature extractor"),
        ("04", "Transfer learning", "Backbone initially frozen while training the classification head"),
        ("05", "Fine-tuning", "Last 30 backbone layers selectively unfrozen; Batch Normalization layers kept frozen"),
        ("06", "Global Average Pooling", "Converts feature maps into a feature vector"),
        ("07", "Dropout", "Rate = 0.3 to reduce overfitting"),
        ("08", "Output layer", "4 class scores with softmax activation"),
    ]

    for number, name, description in architecture:
        html(
            f"""
<div class="panel" style="padding:15px 20px;">
<div style="display:flex;gap:18px;align-items:center;">
<div style="color:#70e1f5;font-weight:700;min-width:25px;">{number}</div>
<div style="flex:1;min-width:0;">
<div style="font-weight:700;color:#f2f6ff;">{name}</div>
<div class="muted">{description}</div>
</div>
</div>
</div>
"""
        )

    st.markdown("---")
    section_heading("Training configuration")

    a, b = st.columns(2, gap="medium")
    with a:
        metric_card("Base architecture", "EfficientNetB0", "Pretrained on ImageNet")
        metric_card("Loss function", "Sparse categorical crossentropy",
                    "For integer-encoded class labels")
    with b:
        metric_card("Input dimensions", "224 × 224", "3 RGB channels")
        metric_card("Fine-tuning optimizer", "Adam · 0.00001",
                    "Low learning rate for fine-tuning")

    st.markdown("---")
    html(
        """
<div class="panel">
<div class="eyebrow">IMPORTANT LIMITATIONS</div>
<h3>Research, not diagnosis</h3>
<div class="muted">
The model predicts among the classes represented in its training data.
It may perform poorly on images from other scanners, hospitals,
acquisition protocols or populations. Softmax scores are not
necessarily calibrated probabilities. Clinical use would require
extensive independent validation and medical oversight.
</div>
</div>
"""
    )

    st.caption(
        "Project: Brain Tumor Classification Using EfficientNetB0 · "
        "Dataset: BRISC 2025"
    )
