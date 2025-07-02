import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

#  Local path to your model
MODEL_PATH = https://drive.google.com/file/d/1R08mOGPySz174lsbmb5J5W52HTeegP_L/view?usp=drive_link

#  Load your trained model
@st.cache_resource
def load_model():
    try:
        model = joblib.load(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None

model = load_model()


st.set_page_config(page_title="CO₂ Emission Predictor", layout="centered")
st.title("🌍 CO₂ Emission Predictor (Local Model)")
st.markdown("Predict total CO₂ emissions using economic and environmental inputs.")


if model:
    st.subheader("🔧 Enter Input Features")

    cereal_yield = st.slider("Cereal yield (kg/ha)", 0, 10000, 3000)
    fdi_perc_gdp = st.slider("FDI (% of GDP)", 0.0, 50.0, 5.0)
    en_per_gdp = st.slider("Energy per GDP", 0.0, 1000.0, 300.0)
    en_per_cap = st.slider("Energy per capita", 0.0, 10000.0, 500.0)
    gdp = st.number_input("GDP (USD)", min_value=0.0, value=1e10)
    pop = st.number_input("Population", min_value=0.0, value=5e7)
    urb_pop_growth_perc = st.slider("Urban population growth (%)", 0.0, 10.0, 2.5)

    # Match feature names used during model training
    input_df = pd.DataFrame({
        'cereal_yield': [cereal_yield],
        'fdi_perc_gdp': [fdi_perc_gdp],
        'en_per_gdp': [en_per_gdp],
        'en_per_cap': [en_per_cap],
        'gdp': [gdp],
        'pop': [pop],
        'urb_pop_growth_perc': [urb_pop_growth_perc]
    })

    with st.expander("📊 View Input Summary"):
        st.dataframe(input_df.T.rename(columns={0: "Value"}))

    if st.button("🔍 Predict CO₂ Emissions"):
        try:
            result = model.predict(input_df)[0]
            st.success(f"🔋 Predicted CO₂ Emissions: **{result:,.2f} million tonnes**")

            if hasattr(model, "feature_importances_"):
                st.write("### 📈 Feature Importance")
                fig, ax = plt.subplots()
                ax.barh(input_df.columns, model.feature_importances_)
                ax.set_xlabel("Importance")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Prediction failed: {e}")
