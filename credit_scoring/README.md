# Credit Scoring Model — CodeAlpha Task 1

## Objective
Build a classification model that estimates credit application status using income, debt, payment history and related features.

## Tech stack
Python, Pandas, Scikit-learn, Random Forest, Streamlit.

## Dataset
`credit_data.csv` is a **synthetic educational dataset** created for this project. It is not real customer or banking data.

## Run locally
```bash
pip install -r requirements.txt
python train_model.py
streamlit run app.py
```

## Features
- Data loading and train/test split
- Missing-value handling with an imputer
- Random Forest classification
- Accuracy, classification report and confusion matrix
- Streamlit form for interactive predictions

## Limitation
This is a learning project, not a credit bureau score or a real lending decision system.
