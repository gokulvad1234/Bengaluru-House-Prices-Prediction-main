from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.joblib"

st.set_page_config(
    page_title="Bengaluru House Price Predictor",
    page_icon="🏠",
    layout="centered",
)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH.name}. "
            "Keep house_price_model.joblib in the same folder as app.py."
        )
    artifacts = joblib.load(MODEL_PATH)
    return artifacts["model"], artifacts["columns"]


try:
    model, columns = load_artifacts()
except Exception as exc:
    st.error(f"Unable to load the trained model: {exc}")
    st.stop()

locations = sorted(
    column for column in columns
    if column not in {"total_sqft", "bath", "bhk"}
)

st.title("🏠 Bengaluru House Price Predictor")
st.write(
    "Estimate a property's asking price using its location, area, "
    "number of bedrooms and bathrooms."
)
st.info(
    "This demo uses a Linear Regression model trained on cleaned "
    "Bengaluru property listings. The estimate is not a formal valuation."
)

location = st.selectbox(
    "Location",
    locations,
    index=locations.index("Whitefield") if "Whitefield" in locations else 0,
)
sqft = st.number_input(
    "Total area (sq ft)", min_value=300, max_value=10000, value=1200, step=50
)
col1, col2 = st.columns(2)
with col1:
    bhk = st.number_input("BHK (bedrooms)", min_value=1, max_value=10, value=2, step=1)
with col2:
    bath = st.number_input("Bathrooms", min_value=1, max_value=10, value=2, step=1)

if st.button("Predict price", type="primary", use_container_width=True):
    values = {column: 0 for column in columns}
    values["total_sqft"] = float(sqft)
    values["bath"] = float(bath)
    values["bhk"] = float(bhk)
    values[location] = 1

    input_data = pd.DataFrame([values], columns=columns)
    prediction = float(model.predict(input_data)[0])

    if not np.isfinite(prediction) or prediction <= 0:
        st.warning(
            "The model returned an unreliable estimate for this combination. "
            "Try a different set of property details."
        )
    else:
        st.success(f"Estimated price: ₹{prediction:,.2f} Lakhs")
        st.caption(f"Approximately ₹{prediction / 100:,.2f} Crore")

st.divider()
st.caption(
    "Model inputs: location, total area, BHK and bathrooms. "
    "Predictions are estimates based on historical asking prices and may differ "
    "from actual transaction values."
)
