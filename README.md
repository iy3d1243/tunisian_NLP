# Arabic Complaint Classifier
try it now : https://tunisiannlp-vft2qtepl3q4zqdtdcu8ba.streamlit.app
A machine learning application that classifies Arabic complaints using various classification models.

## Project Overview

This application uses Natural Language Processing (NLP) techniques to classify Arabic complaints into different categories. It includes:

- Text preprocessing for Arabic language
- Multiple machine learning models:
  - Logistic Regression
  - Naive Bayes
  - Support Vector Machine (SVM)
  - Random Forest
- A Streamlit web interface for easy interaction

## Files in the Project

- `app.py`: Streamlit web application for the classifier
- `models.py`: Script for training and saving the machine learning models
- `*.pkl`: Saved models and vectorizer
- `requirements.txt`: List of required Python packages

## How to Run

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. To retrain the models, run:
   ```
   python models.py
   ```

## Features

- Arabic text preprocessing
- Multiple classification models
- Interactive web interface
- Sample text generation
- Model performance information

## Requirements

- Python 3.6+
- Streamlit
- Scikit-learn
- Pandas
- XGBoost
