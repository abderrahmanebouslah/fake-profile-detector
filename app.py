import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost

# --- Page Configuration ---
st.set_page_config(
    page_title="Fake Profile Detector",
    page_icon="🤖",
    layout="wide", # Use wide layout for better spacing
    initial_sidebar_state="auto"
)

# --- Load Model, Scaler, and Columns ---
@st.cache_resource
def load_resources():
    """Loads model, scaler, and columns once and caches them."""
    try:
        with open('xgb_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        return model, scaler, feature_columns
    except FileNotFoundError:
        st.error("Error: Model or dependency files not found. Please ensure 'xgb_model.pkl', 'scaler.pkl', and 'feature_columns.pkl' are in the same directory as this app.py file.")
        st.stop()
    except Exception as e:
        st.error(f"An error occurred while loading files: {e}")
        st.stop()

model, scaler, feature_columns = load_resources()

# --- Feature Definitions (with ranges and descriptions) ---
feature_definitions = {
    'pos': {'label': 'Number of Posts', 'min': 0, 'max': 5000, 'step': 1},
    'flw': {'label': 'Number of Followers', 'min': 0, 'max': 100000, 'step': 1},
    'flg': {'label': 'Number of Following', 'min': 0, 'max': 10000, 'step': 1},
    'bl': {'label': 'Bio Length (characters)', 'min': 0, 'max': 150, 'step': 1},
    'lt': {'label': 'Account Lifetime (days)', 'min': 0, 'max': 5000, 'step': 1},
    'pic': {'label': 'Has Profile Picture', 'options': [1, 0]},
    'lin': {'label': 'Has Link in Bio', 'options': [1, 0]},
    'cl': {'label': 'Comments per Like', 'min': 0.0, 'max': 100.0, 'step': 0.01},
    'cz': {'label': 'Ratio of Posts with Zero Comments', 'min': 0.0, 'max': 1.0, 'step': 0.01},
    'ni': {'label': 'Ratio of Non-Informative Posts', 'min': 0.0, 'max': 1.0, 'step': 0.01},
    'erl': {'label': 'Engagement Rate per Like', 'min': 0.0, 'max': 50.0, 'step': 0.1},
    'erc': {'label': 'Engagement Rate per Comment', 'min': 0.0, 'max': 5.0, 'step': 0.1},
    'hc': {'label': 'Average Hashtag Count', 'min': 0.0, 'max': 30.0, 'step': 0.1},
    'pr': {'label': 'Promotion Rate', 'min': 0.0, 'max': 1.0, 'step': 0.01},
    'fo': {'label': 'Follower Growth Rate', 'min': 0.0, 'max': 10.0, 'step': 0.01},
    'cs': {'label': 'Content Similarity Score', 'min': 0.0, 'max': 1.0, 'step': 0.01},
    'pi': {'label': 'Profile Information Score', 'min': 0.0, 'max': 1000.0, 'step': 1.0}
}

# --- Callback function to sync widget states ---
def sync_widgets(source_key, target_key):
    st.session_state[target_key] = st.session_state[source_key]

# --- Initialize Session State ---
def initialize_state():
    defaults = {
        "pos": 100, "flw": 500, "flg": 500, "bl": 25, "lt": 365, "pic": 1, "lin": 1,
        "cl": 0.1, "cz": 0.2, "ni": 0.1, "erl": 5.0, "erc": 0.5, "hc": 2.0, "pr": 0.0,
        "fo": 0.1, "cs": 0.3, "pi": 50.0
    }
    for key, value in defaults.items():
        if f"{key}_main" not in st.session_state:
            st.session_state[f"{key}_main"] = value
        if f"{key}_slider" not in st.session_state:
            st.session_state[f"{key}_slider"] = value

initialize_state()

# --- App Header ---
st.title("Social Media Fake Profile Detector 🤖")
st.markdown("""
This app uses a machine learning model to predict if a social media profile is real or fake.
Adjust the details using the inputs below. The sidebar sliders and main area inputs are synchronized.
""")
st.write("---")

# --- User Input Section ---
st.sidebar.header("Profile Features (Sliders)")
st.subheader("Profile Details")

col1, col2, col3 = st.columns(3)

# Define column assignments
col_map = {
    'Account Stats': (col1, ['pos', 'flw', 'flg', 'bl', 'lt']),
    'Engagement Metrics': (col2, ['cl', 'cz', 'ni', 'erl', 'erc']),
    'Other Metrics & Settings': (col3, ['hc', 'pr', 'fo', 'cs', 'pi']),
}

# Dynamically create inputs in columns and sidebar
for group, (column, keys) in col_map.items():
    with column:
        st.write(f"**{group}**")
        for key in keys:
            props = feature_definitions[key]
            label = f"{props['label']} ({key})"
            st.number_input(label, min_value=props['min'], max_value=props['max'], step=props['step'], key=f"{key}_main", on_change=sync_widgets, args=(f"{key}_main", f"{key}_slider"))
            st.sidebar.slider(label, min_value=props['min'], max_value=props['max'], step=props['step'], key=f"{key}_slider", on_change=sync_widgets, args=(f"{key}_slider", f"{key}_main"))

st.write("---")
st.write("**Profile Settings**")
c1, c2 = st.columns(2)
with c1:
    props = feature_definitions['pic']
    st.selectbox(f"{props['label']} (pic)", props['options'], help="1 for Yes, 0 for No", key="pic_main", on_change=sync_widgets, args=("pic_main", "pic_slider"))
    st.sidebar.selectbox(f"{props['label']} (pic)", props['options'], key="pic_slider", on_change=sync_widgets, args=("pic_slider", "pic_main"))
with c2:
    props = feature_definitions['lin']
    st.selectbox(f"{props['label']} (lin)", props['options'], help="1 for Yes, 0 for No", key="lin_main", on_change=sync_widgets, args=("lin_main", "lin_slider"))
    st.sidebar.selectbox(f"{props['label']} (lin)", props['options'], key="lin_slider", on_change=sync_widgets, args=("lin_slider", "lin_main"))


# --- Prediction Logic ---
if st.button('**Predict Profile Authenticity**', type="primary"):
    try:
        # Create DataFrame from session state
        input_data = {key: [st.session_state[f"{key}_main"]] for key, props in feature_definitions.items()}
        input_df = pd.DataFrame(input_data)

        # Feature Engineering
        input_df['follower_following_ratio'] = input_df['flw'] / (input_df['flg'] + 1)
        input_df['engagement_sum'] = input_df['erl'] + input_df['erc']
        input_df['post_frequency'] = input_df['pos'] / (input_df['lt'] + 1)

        # Ensure correct column order
        input_df = input_df[feature_columns]

        # Scale features
        input_scaled = scaler.transform(input_df)

        # Prediction
        prediction = model.predict(input_scaled)
        prediction_proba = model.predict_proba(input_scaled)

        st.subheader("Prediction Result")
        st.write("---")

        if prediction[0] == 1:
            st.error("🔴 This profile is likely **FAKE**.")
        else:
            st.success("🟢 This profile is likely **REAL**.")

        st.subheader("Prediction Confidence Score")
        proba_df = pd.DataFrame(prediction_proba, columns=['Confidence (Real)', 'Confidence (Fake)'])
        st.dataframe(proba_df.style.format("{:.2%}"))

       # st.info("""
       # **How to interpret the score:** The confidence score shows the model's certainty.
       # A score closer to 100% for a class indicates higher confidence in that prediction.
       # """, icon="ℹ️")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# --- Footer ---
st.markdown("---")
st.markdown("[Bslh Abderrahman ♟]")

# python -m pip install streamlit pandas numpy scikit-learn xgboost
# streamlit run app.py