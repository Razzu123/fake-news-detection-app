# Fake News Detection Streamlit App

This folder contains the Streamlit interface for the WELFake dissertation
project. It uses the trained DistilBERT model and reports the genuine held-out
test results for Logistic Regression, Linear SVM and DistilBERT.

## Required folder structure

```text
Fake_News_Detection_App/
├── app.py
├── requirements.txt
├── README.md
├── distilbert_final_model/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer files
└── results/
    ├── distilbert_roc_curve.png
    └── distilbert_shap_explanation.html
```

Copy `distilbert_final_model` and `results` from the downloaded
`Fake_News_Dissertation` folder. Do not copy `distilbert_checkpoints` into the
website folder.

## Run locally on a Mac with Anaconda

Open Terminal and run:

```bash
cd ~/Documents/Fake_News_Detection_App
conda create -n fake-news-app python=3.11 -y
conda activate fake-news-app
pip install -r requirements.txt
streamlit run app.py
```

The local application normally opens at `http://localhost:8501`. This local
address works only on your computer.

## Public deployment

The trained model is too large for a normal GitHub upload. Upload the contents
of `distilbert_final_model` to a Hugging Face model repository. Set this value
in the Streamlit Community Cloud secrets:

```toml
MODEL_REPO_ID = "YOUR_HUGGINGFACE_USERNAME/YOUR_MODEL_REPOSITORY"
```

If the model repository is private, also add:

```toml
HF_TOKEN = "YOUR_HUGGINGFACE_READ_TOKEN"
```

Never put an access token directly in `app.py`, `README.md`, or a public GitHub
repository.

Upload `app.py`, `requirements.txt`, `README.md`, `.gitignore`, `.streamlit`,
and the small `results` files to GitHub. Do not upload the local model folder or
training checkpoints. Connect the GitHub repository to Streamlit Community
Cloud to obtain a public `streamlit.app` address.

## Responsible-use statement

This system identifies linguistic patterns associated with the training data.
It is a decision-support tool and should not replace professional fact-checking.
