import streamlit as st
import pandas as pd
import joblib

# Load model and features
model = joblib.load("forecasting_co2_emmision_krish.pkl")
features = ['cereal_yield', 'gni_per_cap', 'en_per_cap',
            'pop_urb_aggl_perc', 'prot_area_perc', 'gdp', 
            'urb_pop_growth_perc']

# Feature descriptions and safe value ranges
feature_info = {
    'cereal_yield': {
        'desc': 'Cereal yield (kg per hectare of harvested land)',
        'safe': 'Safe range: 2500–6000 kg/ha'
    },
    'gni_per_cap': {
        'desc': 'Gross national income per capita (USD)',
        'safe': 'Safe range: 3000–15000 USD'
    },
    'en_per_cap': {
        'desc': 'Energy use per capita (kg of oil equivalent)',
        'safe': 'Safe range: 1000–4000 kg'
    },
    'pop_urb_aggl_perc': {
        'desc': 'Urban population in agglomerations over 1 million (%)',
        'safe': 'Safe range: 10–80%'
    },
    'prot_area_perc': {
        'desc': 'Protected areas (% of total land area)',
        'safe': 'Safe range: 10–35%'
    },
    'gdp': {
        'desc': 'Gross Domestic Product (in billion USD)',
        'safe': 'Safe range: 50–2500 billion'
    },
    'urb_pop_growth_perc': {
        'desc': 'Urban population growth rate (%)',
        'safe': 'Safe range: 1–5%'
    }
}

# Set page config
st.set_page_config(page_title="CO₂ Emissions Predictor", page_icon="🌍", layout="centered")

# Header
st.title("🌍 CO₂ Emissions Predictor using Machine Learning")
st.markdown("Predict **CO₂ emissions per capita (metric tons)** by entering country-level development indicators.")
st.markdown("---")

# Input fields
inputs = {}
with st.form("input_form"):
    st.subheader("🔢 Enter Indicators")
    for feature in features:
        col1, col2 = st.columns([2, 3])
        with col1:
            inputs[feature] = st.number_input(
                f"{feature.replace('_', ' ').title()}",
                value=0.0,
                help=f"{feature_info[feature]['desc']} | {feature_info[feature]['safe']}"
            )
        with col2:
            st.markdown(f"**{feature_info[feature]['desc']}**  \n:green[{feature_info[feature]['safe']}]")

    submitted = st.form_submit_button("🚀 Predict CO₂ per Capita")

# Prediction and Output
if submitted:
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]

    st.success(f"🌿 Estimated CO₂ per Capita: **{prediction:.2f} metric tons**")
    st.markdown("---")
    st.info("📊 **Tip:** CO₂ emissions per capita are strongly influenced by energy consumption and GDP. To reduce emissions, consider improving energy efficiency and increasing green spaces.")

# Footer
st.markdown("🔬 *Model trained on global indicators and country-level data.*")
st.markdown("💡 *Created by Krish Sharma*")

