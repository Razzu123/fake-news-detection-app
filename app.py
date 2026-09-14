from __future__ import annotations

import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


APP_DIR = Path(__file__).resolve().parent
RESULTS_DIR = APP_DIR / "results"
MAX_LENGTH = 256
MAX_BATCH_ROWS = 100

DISCLAIMER = (
    "This system identifies linguistic patterns associated with the training data. "
    "It is a decision-support tool and should not replace professional fact-checking."
)

MODEL_RESULTS = pd.DataFrame(
    {
        "Model": ["Logistic Regression", "Linear SVM", "DistilBERT"],
        "Accuracy": [0.9596, 0.9663, 0.9924],
        "Macro F1": [0.9593, 0.9661, 0.9923],
        "ROC-AUC": [0.9926, 0.9948, 0.9997],
        "Incorrect Predictions": [515, 429, 97],
    }
)

CONFUSION_MATRICES = {
    "Logistic Regression": np.array([[6651, 307], [208, 5569]]),
    "Linear SVM": np.array([[6704, 254], [175, 5602]]),
    "DistilBERT": np.array([[6935, 23], [74, 5703]]),
}


st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        :root {
            --ink: #102a43;
            --muted: #52667a;
            --teal: #0f766e;
            --teal-soft: #e6fffa;
            --amber: #b45309;
            --amber-soft: #fff7ed;
            --line: #d9e2ec;
        }
        .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero {
            padding: 1.8rem 2rem;
            border-radius: 18px;
            color: white;
            background: linear-gradient(120deg, #0f766e 0%, #155e75 100%);
            box-shadow: 0 14px 34px rgba(15, 118, 110, 0.18);
            margin-bottom: 1.6rem;
        }
        .hero h1 {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3.2rem);
            line-height: 1.05;
        }
        .hero p {
            margin: 0.75rem 0 0;
            max-width: 760px;
            font-size: 1.05rem;
            opacity: 0.94;
        }
        .prediction-card {
            padding: 1.25rem 1.4rem;
            border-radius: 14px;
            margin: 0.8rem 0 1rem;
            border: 1px solid var(--line);
        }
        .prediction-real {
            background: var(--teal-soft);
            border-left: 7px solid var(--teal);
        }
        .prediction-fake {
            background: var(--amber-soft);
            border-left: 7px solid var(--amber);
        }
        .prediction-label {
            font-size: 1.65rem;
            font-weight: 750;
            color: var(--ink);
        }
        .prediction-note {
            color: var(--muted);
            margin-top: 0.25rem;
        }
        .small-card {
            padding: 1rem 1.1rem;
            border: 1px solid var(--line);
            border-radius: 12px;
            background: #ffffff;
            min-height: 112px;
        }
        .small-card strong {
            color: var(--ink);
        }
        .footer {
            border-top: 1px solid var(--line);
            margin-top: 2rem;
            padding-top: 1rem;
            color: var(--muted);
            font-size: 0.88rem;
        }
        div[data-testid="stMetric"] {
            border: 1px solid var(--line);
            padding: 0.8rem 1rem;
            border-radius: 12px;
            background: white;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


def normalize_text(value: object) -> str:
    """Convert an input value to clean, single-spaced text."""
    if value is None or pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def get_optional_setting(name: str) -> str:
    """Read a deployment setting without failing when no secrets file exists."""
    environment_value = os.getenv(name, "").strip()
    if environment_value:
        return environment_value

    try:
        secret_value = st.secrets.get(name, "")
    except (FileNotFoundError, KeyError):
        return ""
    return str(secret_value).strip() if secret_value else ""


def find_model_source() -> str:
    """Prefer the downloaded local model, then fall back to a Hub repository."""
    local_candidates = [
        APP_DIR / "distilbert_final_model",
        APP_DIR / "Fake_News_Dissertation" / "distilbert_final_model",
    ]

    for candidate in local_candidates:
        if (candidate / "config.json").exists():
            return str(candidate)

    return get_optional_setting("MODEL_REPO_ID")


@st.cache_resource(show_spinner=False)
def load_model_resources(model_source: str, access_token: str):
    """Load the tokenizer and trained classifier once per app session."""
    authentication = {"token": access_token} if access_token else {}
    tokenizer = AutoTokenizer.from_pretrained(model_source, **authentication)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source,
        **authentication,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device


def require_model():
    """Load the model or show a clear setup message."""
    model_source = find_model_source()
    if not model_source:
        st.error(
            "The trained model was not found. Copy the `distilbert_final_model` "
            "folder beside `app.py`, or configure `MODEL_REPO_ID` for deployment."
        )
        st.stop()

    try:
        with st.spinner("Loading the trained DistilBERT model..."):
            return load_model_resources(
                model_source,
                get_optional_setting("HF_TOKEN"),
            )
    except Exception as error:
        st.error("The trained model could not be loaded.")
        with st.expander("Technical details"):
            st.code(str(error))
        st.stop()


def predict_probabilities(texts: list[str]) -> np.ndarray:
    """Return Fake and Real probabilities for one or more texts."""
    tokenizer, model, device = require_model()
    all_probabilities: list[np.ndarray] = []

    for start in range(0, len(texts), 16):
        batch = texts[start : start + 16]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
            if key in {"input_ids", "attention_mask"}
        }

        with torch.no_grad():
            logits = model(**encoded).logits
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        all_probabilities.append(probabilities)

    return np.vstack(all_probabilities)


def render_header() -> None:
    st.markdown(
        """
        <section class="hero">
            <h1>Fake News Detection</h1>
            <p>
                A text-classification prototype using a fine-tuned DistilBERT model
                to identify language patterns associated with fake and real news.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_prediction_result(probabilities: np.ndarray) -> None:
    fake_probability = float(probabilities[0])
    real_probability = float(probabilities[1])
    prediction_id = int(np.argmax(probabilities))
    prediction = "Real" if prediction_id == 1 else "Fake"
    confidence = float(probabilities[prediction_id])
    card_class = "prediction-real" if prediction == "Real" else "prediction-fake"

    st.markdown(
        f"""
        <div class="prediction-card {card_class}">
            <div class="prediction-label">Prediction: {prediction}</div>
            <div class="prediction-note">
                Model confidence: {confidence * 100:.2f}% — this is not independent
                verification of the article's factual truth.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    metric_one, metric_two, metric_three = st.columns(3)
    metric_one.metric("Predicted class", prediction)
    metric_two.metric("Fake probability", f"{fake_probability * 100:.2f}%")
    metric_three.metric("Real probability", f"{real_probability * 100:.2f}%")

    probability_table = pd.DataFrame(
        {
            "Class": ["Fake", "Real"],
            "Probability": [fake_probability, real_probability],
        }
    ).set_index("Class")
    st.bar_chart(probability_table, color="#0f766e")
    st.caption(
        "Long inputs are truncated to the first 256 model tokens, matching the "
        "maximum length used in this implementation."
    )


def render_single_prediction() -> None:
    st.markdown("#### Analyse one article")
    title = st.text_input(
        "News title (optional)",
        placeholder="Enter the headline",
    )
    article = st.text_area(
        "Article text",
        height=250,
        placeholder="Paste the news article text here...",
    )
    combined_text = normalize_text(f"{title} {article}")
    st.caption(f"{len(combined_text):,} characters entered")

    if st.button("Analyse article", type="primary", use_container_width=True):
        if len(combined_text) < 50:
            st.warning(
                "Please enter at least 50 characters. Very short text does not "
                "provide enough context for a meaningful screening result."
            )
        else:
            with st.spinner("Analysing linguistic patterns..."):
                probabilities = predict_probabilities([combined_text])[0]
            render_prediction_result(probabilities)


def locate_csv_columns(dataframe: pd.DataFrame) -> tuple[str | None, str | None]:
    normalized_columns = {
        str(column).strip().lower(): str(column) for column in dataframe.columns
    }
    title_column = normalized_columns.get("title")
    text_column = None
    for candidate in ("text", "content", "article", "article_text"):
        if candidate in normalized_columns:
            text_column = normalized_columns[candidate]
            break
    return title_column, text_column


def render_batch_prediction() -> None:
    st.markdown("#### Analyse a CSV file")
    st.write(
        "Upload a CSV containing a `text`, `content`, `article`, or `article_text` "
        "column. A `title` column is optional."
    )
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        accept_multiple_files=False,
    )

    if uploaded_file is None:
        return

    try:
        uploaded_data = pd.read_csv(uploaded_file)
    except Exception as error:
        st.error(f"The CSV could not be read: {error}")
        return

    title_column, text_column = locate_csv_columns(uploaded_data)
    if text_column is None:
        st.error(
            "No article-text column was found. Rename the relevant column to "
            "`text` or `content` and upload the file again."
        )
        return

    st.write(f"Uploaded rows: **{len(uploaded_data):,}**")
    st.dataframe(uploaded_data.head(5), use_container_width=True)

    if len(uploaded_data) > MAX_BATCH_ROWS:
        st.info(
            f"For responsive demonstration, only the first {MAX_BATCH_ROWS} "
            "rows will be analysed."
        )

    if not st.button("Analyse CSV", type="primary", use_container_width=True):
        return

    output_data = uploaded_data.head(MAX_BATCH_ROWS).copy()
    titles = (
        output_data[title_column].map(normalize_text)
        if title_column
        else pd.Series("", index=output_data.index)
    )
    articles = output_data[text_column].map(normalize_text)
    combined_texts = [
        normalize_text(f"{title} {article}")
        for title, article in zip(titles, articles)
    ]

    valid_positions = [
        position for position, text in enumerate(combined_texts) if len(text) >= 20
    ]
    output_data["Prediction"] = "Insufficient text"
    output_data["Fake Probability"] = np.nan
    output_data["Real Probability"] = np.nan

    if valid_positions:
        valid_texts = [combined_texts[position] for position in valid_positions]
        with st.spinner("Analysing uploaded articles..."):
            probabilities = predict_probabilities(valid_texts)

        for result_position, row_position in enumerate(valid_positions):
            fake_probability = float(probabilities[result_position, 0])
            real_probability = float(probabilities[result_position, 1])
            output_data.iloc[
                row_position,
                output_data.columns.get_loc("Prediction"),
            ] = "Real" if real_probability >= fake_probability else "Fake"
            output_data.iloc[
                row_position,
                output_data.columns.get_loc("Fake Probability"),
            ] = fake_probability
            output_data.iloc[
                row_position,
                output_data.columns.get_loc("Real Probability"),
            ] = real_probability

    st.success("CSV analysis completed.")
    st.dataframe(output_data, use_container_width=True)
    st.download_button(
        "Download prediction results",
        data=output_data.to_csv(index=False).encode("utf-8"),
        file_name="fake_news_predictions.csv",
        mime="text/csv",
        use_container_width=True,
    )


def render_prediction_page() -> None:
    st.subheader("News classification")
    st.write(
        "Enter an article manually or upload a CSV. Input text is processed for "
        "the current session and is not intentionally stored by this application."
    )
    single_tab, batch_tab = st.tabs(["Single article", "CSV batch"])
    with single_tab:
        render_single_prediction()
    with batch_tab:
        render_batch_prediction()
    st.warning(DISCLAIMER)


def create_comparison_figure():
    score_columns = ["Accuracy", "Macro F1", "ROC-AUC"]
    x_positions = np.arange(len(MODEL_RESULTS))
    width = 0.23
    colors = ["#0f766e", "#0284c7", "#7c3aed"]

    figure, axis = plt.subplots(figsize=(10, 5.6))
    for index, (metric, color) in enumerate(zip(score_columns, colors)):
        offsets = x_positions + (index - 1) * width
        bars = axis.bar(
            offsets,
            MODEL_RESULTS[metric],
            width,
            label=metric,
            color=color,
        )
        axis.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)

    axis.set_xticks(x_positions)
    axis.set_xticklabels(MODEL_RESULTS["Model"])
    axis.set_ylim(0.94, 1.005)
    axis.set_ylabel("Score")
    axis.set_title("Held-out WELFake Test-Set Performance")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(loc="lower right")
    figure.tight_layout()
    return figure


def create_confusion_matrix_figure(model_name: str):
    matrix = CONFUSION_MATRICES[model_name]
    figure, axis = plt.subplots(figsize=(5.2, 4.3))
    image = axis.imshow(matrix, cmap="Blues")

    threshold = matrix.max() / 2
    for row in range(2):
        for column in range(2):
            axis.text(
                column,
                row,
                f"{matrix[row, column]:,}",
                ha="center",
                va="center",
                fontsize=13,
                color="white" if matrix[row, column] > threshold else "#102a43",
            )

    axis.set_xticks([0, 1], labels=["Fake", "Real"])
    axis.set_yticks([0, 1], labels=["Fake", "Real"])
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"{model_name} Confusion Matrix")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    return figure


def find_result_file(filename: str) -> Path | None:
    candidates = [
        RESULTS_DIR / filename,
        APP_DIR / "Fake_News_Dissertation" / "results" / filename,
    ]
    return next((path for path in candidates if path.exists()), None)


def render_performance_page() -> None:
    st.subheader("Model performance")
    st.info(
        "These are genuine results from the same held-out WELFake test set "
        "containing 12,735 deduplicated articles."
    )

    metric_one, metric_two, metric_three, metric_four = st.columns(4)
    metric_one.metric("Best model", "DistilBERT")
    metric_two.metric("Test accuracy", "99.24%")
    metric_three.metric("Macro F1", "99.23%")
    metric_four.metric("ROC-AUC", "99.97%")

    st.markdown("#### Three-model comparison")
    display_results = MODEL_RESULTS.copy()
    for column in ["Accuracy", "Macro F1", "ROC-AUC"]:
        display_results[column] = display_results[column].map(lambda value: f"{value:.4f}")
    st.dataframe(display_results, hide_index=True, use_container_width=True)
    st.pyplot(create_comparison_figure(), use_container_width=True)

    st.markdown("#### Confusion matrices")
    matrix_tabs = st.tabs(list(CONFUSION_MATRICES))
    for tab, model_name in zip(matrix_tabs, CONFUSION_MATRICES):
        with tab:
            st.pyplot(
                create_confusion_matrix_figure(model_name),
                use_container_width=False,
            )

    st.markdown("#### DistilBERT ROC curve")
    roc_curve_path = find_result_file("distilbert_roc_curve.png")
    if roc_curve_path:
        st.image(str(roc_curve_path), use_container_width=True)
    else:
        st.write(
            "DistilBERT achieved an ROC-AUC of **0.9997**. Copy the downloaded "
            "`results` folder beside `app.py` to display the saved ROC-curve image."
        )

    st.markdown("#### Example SHAP explanation")
    st.write(
        "The example explains which tokens pushed one test prediction towards "
        "the Fake or Real class. It is a local explanation, not proof of causality."
    )
    shap_path = find_result_file("distilbert_shap_explanation.html")
    if shap_path:
        components.html(
            shap_path.read_text(encoding="utf-8"),
            height=650,
            scrolling=True,
        )
    else:
        st.caption(
            "Copy the downloaded `results` folder beside `app.py` to display "
            "the interactive SHAP output."
        )

    st.warning(
        "Report these values as performance on the held-out WELFake test set, "
        "not as universal real-world accuracy."
    )


def render_about_page() -> None:
    st.subheader("About this system")
    st.write(
        "This MSc dissertation prototype compares TF-IDF Logistic Regression, "
        "TF-IDF Linear SVM and a fine-tuned DistilBERT classifier. DistilBERT "
        "produced the strongest internal test-set result and is used for the "
        "interactive prediction interface."
    )

    column_one, column_two, column_three = st.columns(3)
    with column_one:
        st.markdown(
            """
            <div class="small-card">
                <strong>1. Text preparation</strong><br>
                The title and article body are combined and tokenized.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with column_two:
        st.markdown(
            """
            <div class="small-card">
                <strong>2. Classification</strong><br>
                DistilBERT estimates probabilities for Fake and Real labels.
            </div>
            """,
            unsafe_allow_html=True,
        )
    with column_three:
        st.markdown(
            """
            <div class="small-card">
                <strong>3. Decision support</strong><br>
                The result supports screening but does not verify external evidence.
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("#### Dataset and evaluation")
    st.markdown(
        """
        - Dataset: WELFake, using article titles and text.
        - Cleaned records after duplicate removal: 63,672.
        - Labels: `0 = Fake` and `1 = Real`.
        - Split: 45,843 training, 5,094 validation and 12,735 testing records.
        - Final model: fine-tuned DistilBERT with a 256-token maximum input length.
        """
    )

    st.markdown("#### Important limitations")
    st.markdown(
        """
        - The model learns correlations in one training dataset; it does not search
          evidence or independently determine factual truth.
        - Source style, topic, publication date and dataset construction may influence
          predictions and may produce an unusually high internal score.
        - Performance may decline for new events, social-media slang, satire,
          multilingual content or changing misinformation strategies.
        - False classifications can harm trust, so predictions require human review
          and comparison with reliable sources.
        """
    )
    st.warning(DISCLAIMER)


def render_sidebar() -> str:
    st.sidebar.title("Project navigation")
    page = st.sidebar.radio(
        "Select a section",
        ["Predict News", "Model Performance", "About & Limitations"],
        label_visibility="collapsed",
    )
    st.sidebar.divider()
    st.sidebar.metric("Best test accuracy", "99.24%")
    st.sidebar.caption("Best evaluated model: DistilBERT")
    st.sidebar.caption("Dataset: WELFake")
    return page


def render_footer() -> None:
    st.markdown(
        """
        <div class="footer">
            MSc Computer Science dissertation prototype · Text-based decision support
            for fake-news screening
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    page = render_sidebar()
    render_header()

    if page == "Predict News":
        render_prediction_page()
    elif page == "Model Performance":
        render_performance_page()
    else:
        render_about_page()

    render_footer()


if __name__ == "__main__":
    main()
