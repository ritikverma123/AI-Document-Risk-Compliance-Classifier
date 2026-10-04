import io, re, os
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def extract_text_from_file(uploaded_file):
    """Extract text from PDF, DOCX, TXT, MD or CSV."""
    name = uploaded_file.name.lower()
    data = uploaded_file.read()

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(io.BytesIO(data))
            return "\n".join((page.extract_text() or "") for page in reader.pages)
        except Exception as e:
            raise ValueError(f"Could not read PDF: {e}")

    if name.endswith(".docx"):
        try:
            from docx import Document
            doc = Document(io.BytesIO(data))
            return "\n".join(p.text for p in doc.paragraphs)
        except Exception as e:
            raise ValueError(f"Could not read DOCX: {e}")

    if name.endswith((".txt", ".md", ".csv")):
        return data.decode("utf-8", errors="ignore")

    raise ValueError("Unsupported file type. Upload PDF, DOCX, TXT, MD or CSV.")

def clean_text(text):
    text = re.sub(r"\s+", " ", text or "")
    text = re.sub(r"[^\w\s@./:-]", " ", text)
    return text.strip()

def classify_document(text, vectorizer, model, threshold=0.60):
    cleaned = clean_text(text)
    X = vectorizer.transform([cleaned])
    probabilities = model.predict_proba(X)[0]
    classes = model.classes_
    idx = int(np.argmax(probabilities))
    predicted = classes[idx]
    confidence = float(probabilities[idx])

    # Low-confidence cases are sent to manual review.
    review = confidence < threshold

    return {
        "prediction": predicted,
        "confidence": confidence,
        "manual_review": review,
        "probabilities": {str(c): float(p) for c, p in zip(classes, probabilities)},
        "features": X
    }

def important_terms(text, vectorizer, model, predicted_class, top_n=10):
    X = vectorizer.transform([clean_text(text)])
    feature_names = np.array(vectorizer.get_feature_names_out())
    row = X.toarray()[0]

    if hasattr(model, "coef_"):
        class_index = list(model.classes_).index(predicted_class)
        if len(model.classes_) == 2:
            weights = model.coef_[0] if class_index == 1 else -model.coef_[0]
        else:
            weights = model.coef_[class_index]
        scores = row * weights
    else:
        scores = row

    indices = np.argsort(scores)[::-1]
    terms = []
    for i in indices:
        if row[i] > 0 and scores[i] > 0:
            terms.append((feature_names[i], float(scores[i])))
        if len(terms) >= top_n:
            break
    return terms

def similar_documents(text, vectorizer, training_df, top_n=5):
    X_all = vectorizer.transform(training_df["text"].fillna("").map(clean_text))
    X_query = vectorizer.transform([clean_text(text)])
    sims = cosine_similarity(X_query, X_all)[0]
    idxs = np.argsort(sims)[::-1][:top_n]
    out = training_df.iloc[idxs].copy()
    out["similarity"] = sims[idxs]
    return out[["document_id", "label", "similarity", "text"]]
