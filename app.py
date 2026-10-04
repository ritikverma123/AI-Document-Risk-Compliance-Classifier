import os, pickle
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

from src.utils import (
    extract_text_from_file, clean_text, classify_document,
    important_terms, similar_documents
)

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE, "data", "documents.csv")
MODEL_PATH = os.path.join(BASE, "models", "risk_classifier.pkl")
VECT_PATH = os.path.join(BASE, "models", "tfidf_vectorizer.pkl")

st.set_page_config(page_title="AI Document Risk & Compliance Classifier", page_icon="🛡️", layout="wide")

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    with open(VECT_PATH, "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def evaluate(model, vectorizer, df):
    X = vectorizer.transform(df["text"].fillna("").map(clean_text))
    y = df["label"]
    pred = model.predict(X)
    acc = accuracy_score(y, pred)
    p, r, f1, _ = precision_recall_fscore_support(y, pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y, pred, labels=model.classes_)
    return acc, p, r, f1, cm, pred

st.title("🛡️ AI Document Risk & Compliance Classifier")
st.caption("Automatically classify business documents into risk/compliance categories and flag documents that may need manual review.")

if not (os.path.exists(MODEL_PATH) and os.path.exists(VECT_PATH)):
    st.error("Model files are missing. Run: python train_model.py")
    st.stop()

model, vectorizer = load_model()
df = load_data()

with st.sidebar:
    st.header("⚙️ Review Settings")
    threshold = st.slider(
        "Manual-review confidence threshold",
        min_value=0.50, max_value=0.95, value=0.60, step=0.05,
        help="Predictions below this confidence are flagged for manual review."
    )
    st.divider()
    st.write("**Categories**")
    for c in model.classes_:
        st.write("•", c)
    st.divider()
    st.write("**Supported files:** PDF, DOCX, TXT, MD, CSV")

tab1, tab2, tab3 = st.tabs(["📄 Classify Document", "📊 Dashboard & Evaluation", "📚 Training Data"])

with tab1:
    st.subheader("Upload a business document")
    uploaded = st.file_uploader("Upload document", type=["pdf", "docx", "txt", "md", "csv"])

    manual_text = st.text_area(
        "Or paste document text",
        height=180,
        placeholder="Paste document text here..."
    )

    if uploaded is not None or manual_text.strip():
        try:
            raw_text = extract_text_from_file(uploaded) if uploaded else manual_text
            cleaned = clean_text(raw_text)

            if not cleaned:
                st.warning("No readable text was found.")
                st.stop()

            result = classify_document(cleaned, vectorizer, model, threshold)

            c1, c2, c3 = st.columns(3)
            c1.metric("Predicted Category", result["prediction"])
            c2.metric("Confidence", f"{result['confidence']:.1%}")
            c3.metric("Manual Review", "YES" if result["manual_review"] else "NO")

            if result["manual_review"]:
                st.warning("⚠️ Low-confidence case: this document should go to manual review.")
            else:
                st.success("✅ Confidence is above the manual-review threshold.")

            st.subheader("Category probabilities")
            prob_df = pd.DataFrame(
                [{"Category": k, "Probability": v} for k, v in result["probabilities"].items()]
            ).sort_values("Probability", ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)

            st.subheader("Explainability: important terms")
            terms = important_terms(cleaned, vectorizer, model, result["prediction"])
            if terms:
                st.write(", ".join(f"**{t}**" for t, _ in terms))
                st.dataframe(
                    pd.DataFrame(terms, columns=["Term", "Contribution"]).round(4),
                    use_container_width=True, hide_index=True
                )
            else:
                st.info("No strong positive terms were found in the document.")

            st.subheader("Nearest similar training documents")
            sims = similar_documents(cleaned, vectorizer, df, top_n=5)
            sims["similarity"] = sims["similarity"].map(lambda x: f"{x:.1%}")
            st.dataframe(sims, use_container_width=True, hide_index=True)

            with st.expander("Extracted / cleaned text"):
                st.write(cleaned)

        except Exception as e:
            st.error(str(e))
    else:
        st.info("Upload a document or paste text to start classification.")

with tab2:
    st.subheader("Class-wise model performance")
    acc, precision, recall, f1, cm, pred = evaluate(model, vectorizer, df)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Accuracy", f"{acc:.1%}")
    m2.metric("Weighted Precision", f"{precision:.1%}")
    m3.metric("Weighted Recall", f"{recall:.1%}")
    m4.metric("Weighted F1", f"{f1:.1%}")

    metrics_rows = []
    for cls in model.classes_:
        mask = df["label"] == cls
        p, r, f, support = precision_recall_fscore_support(
            df["label"], pred, labels=[cls], average=None, zero_division=0
        )
        metrics_rows.append([cls, p[0], r[0], f[0], int(mask.sum())])

    metrics_df = pd.DataFrame(
        metrics_rows, columns=["Class", "Precision", "Recall", "F1 Score", "Support"]
    )
    st.dataframe(
        metrics_df.style.format({"Precision":"{:.1%}", "Recall":"{:.1%}", "F1 Score":"{:.1%}"}),
        use_container_width=True, hide_index=True
    )

    st.subheader("Confusion Matrix")
    cm_df = pd.DataFrame(cm, index=model.classes_, columns=model.classes_)
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Flagged-document simulation")
    st.caption(f"Documents with confidence below {threshold:.0%} are considered manual-review cases.")
    X = vectorizer.transform(df["text"].fillna("").map(clean_text))
    probs = model.predict_proba(X)
    max_probs = probs.max(axis=1)
    flagged = df.copy()
    flagged["predicted_category"] = model.predict(X)
    flagged["confidence"] = max_probs
    flagged["manual_review"] = flagged["confidence"] < threshold
    flagged_view = flagged[flagged["manual_review"]].copy()
    st.metric("Flagged documents in training dataset", len(flagged_view))
    if len(flagged_view):
        flagged_view["confidence"] = flagged_view["confidence"].map(lambda x: f"{x:.1%}")
        st.dataframe(
            flagged_view[["document_id","label","predicted_category","confidence","text"]],
            use_container_width=True, hide_index=True
        )
    else:
        st.success("No documents fall below the selected threshold.")

with tab3:
    st.subheader("Sample training documents")
    st.write(f"Total training examples: **{len(df)}**")
    st.dataframe(df, use_container_width=True, hide_index=True)
