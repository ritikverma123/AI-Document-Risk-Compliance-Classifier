# AI Document Risk & Compliance Classifier

This project follows the six requested steps exactly:

1. **Collect sample documents and define categories** such as low, medium, high risk, and policy-specific classes.
2. **Extract text from PDFs/documents** and remove irrelevant formatting.
3. **Generate TF-IDF features and train a classifier.**
4. **Add explainability** using important terms and nearest similar documents.
5. **Set a confidence threshold** so uncertain cases go to manual review.
6. **Evaluate class-wise performance and build a dashboard** showing flagged documents.

## Technology stack
- Python
- pandas / NumPy
- scikit-learn
- Streamlit
- pypdf for PDF extraction
- python-docx for DOCX extraction
- Git/GitHub ready

## Project structure
```text
AI_Document_Risk_Compliance_Classifier/
├── app.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── documents.csv
├── models/
│   └── (generated after training)
├── src/
│   ├── __init__.py
│   └── utils.py
└── sample_documents/
    ├── sample_low_risk.txt
    ├── sample_medium_risk.txt
    ├── sample_high_risk.txt
    └── sample_policy.txt
```

## How to run

### 1. Open terminal in the project folder
```bash
cd AI_Document_Risk_Compliance_Classifier
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Train the model
```bash
python train_model.py
```

### 4. Start the dashboard
```bash
streamlit run app.py
```

The browser will open the Streamlit application.

## Classification categories
- Low Risk
- Medium Risk
- High Risk
- Policy-Specific

## Important note
The included dataset is **synthetic demonstration data** for an academic/project prototype. For real compliance use, replace it with properly labeled organizational documents and validate the model with domain experts.
