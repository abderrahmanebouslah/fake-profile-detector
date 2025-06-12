# Social Media Fake Profile Detector

This project uses machine learning to detect fake social media profiles. It includes a complete pipeline for data preprocessing, feature engineering, model training, and evaluation. The final, optimized model is deployed as an interactive web application using Streamlit.

Kaggle Notebook link : `https://www.kaggle.com/code/abderahmanebouslah/pw2-bslh`

discreption of dataset : `https://docs.google.com/spreadsheets/d/13cRQskL-pA06mfVFwu3tNRSUdgclK0IuEuoDUzmpMgU/edit?usp=sharing`

---

## Features

- **Interactive Web App:** A user-friendly interface built with Streamlit to get real-time predictions.
- **Synchronized Inputs:** Users can input profile data using sliders or direct text entry, and the controls stay in sync.
- **Robust Input Validation:** The app prevents users from entering invalid data (e.g., ratios greater than 1.0).
- **Multiple Models Tested:** The project evaluates several models, including Logistic Regression, Random Forest, a Neural Network, and XGBoost.
- **Optimized Champion Model:** The best model (XGBoost) is fine-tuned using GridSearchCV to maximize performance.
- **Automated Feature Engineering:** The app automatically calculates insightful features like follower-to-following ratio from basic inputs.

---

## Final Model Performance

The final, tuned XGBoost model achieved the following performance on the test set:

- **Accuracy:** ~91%
- **AUC-ROC Score:** ~0.965
- **Key Finding:** The model is highly effective at distinguishing between real and fake profiles, with a precision of ~95% (meaning its "fake" predictions are very trustworthy).

---

## How to Run the App Locally

To run the web application on your own machine, please follow these steps:

1.  **Clone the Repository (or Download the ZIP)**
    ```bash
    git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
    cd your-repo-name
    ```

2.  **Install the Required Libraries**
    ```bash
    pip install -r requirements.txt
    python -m pip install streamlit
    
    ```

3.  **Run the Streamlit App**
    ```bash
    streamlit run app.py
    ```
    Your web browser should open with the application running.

---

## Project Structure

├── app.py                  # The Python script for the Streamlit web application.

├── xgb_model.pkl           # The saved, trained XGBoost model file.

├── scaler.pkl              # The saved Scikit-learn StandardScaler object.

├── feature_columns.pkl     # The saved list of feature names.

├── requirements.txt        # A list of all necessary Python packages.

└── README.md               # This file.
---

## Technologies Used

- **Python**
- **Pandas** for data manipulation.
- **Scikit-learn** for preprocessing and modeling.
- **XGBoost** for the champion classification model.
- **Streamlit** for building the interactive web app.
- **Pickle** for model serialization.
