"""PlantVision: Interactive Streamlit Demo for Plant Leaf Disease Classification."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
from PIL import Image
import streamlit as st

# Configure page settings
st.set_page_config(
    page_title="PlantVision — Plant Leaf Disease Classifier",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e293b;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        margin-bottom: 12px;
    }
    .status-healthy {
        background-color: #dcfce7;
        color: #166534;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
    .status-diseased {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 4px 10px;
        border-radius: 6px;
        font-weight: 600;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading PyTorch classification model...")
def get_cached_predictor(checkpoint_path: str):
    """Load and cache the PlantPredictor model in memory."""
    from plantvision.predict import PlantPredictor
    return PlantPredictor(checkpoint_path=checkpoint_path)


def format_class_name(raw_name: str) -> tuple[str, str, bool]:
    """Parse raw class name into (crop, disease, is_healthy)."""
    parts = raw_name.split("___")
    crop = parts[0].replace("_", " ").title()
    disease = parts[1].replace("_", " ").title() if len(parts) > 1 else "Unknown"
    is_healthy = "healthy" in disease.lower()
    return crop, disease, is_healthy


def main():
    st.markdown('<div class="main-title">🌿 PlantVision — Leaf Disease Classifier</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-title">Empirical Computer Vision classification system using PyTorch, '
        'Transfer Learning (ResNet-18), and verified diagnostic analysis.</div>',
        unsafe_allow_html=True,
    )

    # Sidebar: Model configuration and selection
    st.sidebar.header("⚙️ Model Configuration")

    models_dir = Path("models")
    available_checkpoints = list(models_dir.glob("*.pt")) if models_dir.exists() else []

    if not available_checkpoints:
        st.warning(
            "⚠️ No trained model checkpoints found in `models/` directory.\n\n"
            "To train the models, run the following command in your terminal:\n"
            "```bash\n"
            "python scripts/train_transfer_learning.py\n"
            "```"
        )
        selected_checkpoint = None
    else:
        checkpoint_names = [ckpt.name for ckpt in available_checkpoints]
        # Prefer resnet18_transfer_best.pt as default if present
        default_idx = 0
        for i, name in enumerate(checkpoint_names):
            if "resnet" in name.lower():
                default_idx = i
                break

        selected_name = st.sidebar.selectbox("Select Model Checkpoint", checkpoint_names, index=default_idx)
        selected_checkpoint = str(models_dir / selected_name)

    top_k_val = st.sidebar.slider("Top Candidates to Display", min_value=1, max_value=5, value=3)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 About PlantVision")
    st.sidebar.info(
        "PlantVision is an end-to-end Computer Vision project demonstrating deep learning "
        "transfer learning vs. scratch CNN baselines on the PlantVillage dataset."
    )

    # Layout: Two columns (Input & Preview / Results)
    col_input, col_results = st.columns([1, 1], gap="large")

    with col_input:
        st.subheader("1. Input Leaf Image")
        uploaded_file = st.file_uploader(
            "Upload an image of a plant leaf (.jpg, .jpeg, .png)",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear photograph of a single plant leaf.",
        )

        image_to_process = None
        if uploaded_file is not None:
            try:
                image_to_process = Image.open(uploaded_file)
                st.image(image_to_process, caption="Uploaded Leaf Image", use_container_width=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")

        st.markdown("---")
        predict_button = st.button("🔍 Analyze Leaf Image", type="primary", use_container_width=True)

    with col_results:
        st.subheader("2. Diagnostic Analysis")

        if predict_button:
            if image_to_process is None:
                st.warning("Please upload an image before running analysis.")
            elif selected_checkpoint is None:
                st.error("No model checkpoint is available for inference.")
            else:
                try:
                    predictor = get_cached_predictor(selected_checkpoint)
                    result = predictor.predict(image_to_process, top_k=top_k_val)

                    crop, condition, is_healthy = format_class_name(result.top_class)

                    # Top result badge
                    st.markdown("#### Primary Prediction")
                    if is_healthy:
                        badge_html = f'<span class="status-healthy">HEALTHY ({crop})</span>'
                    else:
                        badge_html = f'<span class="status-diseased">DISEASE DETECTED ({condition})</span>'
                    st.markdown(badge_html, unsafe_allow_html=True)

                    st.markdown(f"**Crop Species:** {crop}")
                    st.markdown(f"**Condition / Diagnosis:** {condition}")
                    st.markdown(f"**Model Confidence:** `{result.top_score:.2%}`")
                    st.markdown(f"**Inference Latency:** `{result.inference_latency_ms:.2f} ms` ({result.device})")

                    # Probability distribution bar chart
                    st.markdown("#### Top Candidate Probabilities")
                    chart_data = []
                    for cls_name, score in result.top_k:
                        c, d, _ = format_class_name(cls_name)
                        label = f"{c}: {d}"
                        chart_data.append({"Category": label, "Probability": score})

                    df_chart = pd.DataFrame(chart_data)
                    st.bar_chart(df_chart.set_index("Category"))

                    # Diagnostics note
                    with st.expander("ℹ️ Understanding Softmax Scores"):
                        st.caption(
                            "The probability scores reflect the model's relative output activation normalized via "
                            "Softmax across known training classes. Softmax scores are sensitive to out-of-distribution "
                            "samples and should not be confused with true Bayesian epidemiological certainty."
                        )

                except Exception as exc:
                    st.error(f"Prediction failed: {exc}")
        else:
            st.info("Upload a leaf image and click **Analyze Leaf Image** to generate predictions.")

    # Model Limitations & Disclaimers
    st.markdown("---")
    st.markdown("### ⚠️ Model Limitations and Agricultural Disclaimer")
    st.warning(
        "**Important Advisory:** This model was trained and evaluated on controlled research datasets "
        "(such as the public PlantVillage repository). Real-world field conditions—including complex backgrounds, "
        "variable sunlight, shadows, wet foliage, and multi-disease coinfections—may significantly impact prediction reliability.\n\n"
        "This tool is an educational machine-learning demonstration and is **not a definitive agricultural diagnosis**. "
        "Do not apply chemical treatments, fungicides, or pesticides without consultation from an agronomist or local agricultural extension service."
    )


if __name__ == "__main__":
    main()
