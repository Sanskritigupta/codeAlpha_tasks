# Heart Disease Prediction — CodeAlpha Task 4

## Objective
Demonstrate a classification workflow using health-related features to predict a heart-disease label.

## Tech stack
Python, Pandas, Scikit-learn, Random Forest, Streamlit.

## Dataset
`heart_disease_data.csv` is a **synthetic educational dataset**. It is not clinical data and must not be used to make health decisions.

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

## Medical disclaimer
This model is not a diagnostic tool. A healthcare professional should assess all medical concerns.
