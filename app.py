import gradio as gr
import joblib
import numpy as np

# ===============================
# 1️⃣ Load the trained model locally
# ===============================
model = joblib.load("house_price_model.pkl")  # make sure this file is in the same folder

# ===============================
# 2️⃣ Prediction Function
# ===============================
def predict_house_price(
    living_area_sqft, number_of_bedrooms, number_of_full_bathrooms,
    number_of_half_bathrooms, garage_capacity_cars, year_built,
    lot_area_sqft, overall_quality_rating, overall_condition_rating,
    basement_area_sqft, garage_area_sqft, first_floor_area_sqft,
    second_floor_area_sqft, number_of_fireplaces
):
    features = np.array([[ 
        living_area_sqft,
        number_of_bedrooms,
        number_of_full_bathrooms,
        number_of_half_bathrooms,
        garage_capacity_cars,
        year_built,
        lot_area_sqft,
        overall_quality_rating,
        overall_condition_rating,
        basement_area_sqft,
        garage_area_sqft,
        first_floor_area_sqft,
        second_floor_area_sqft,
        number_of_fireplaces
    ]])
    
    prediction = model.predict(features)[0]
    return f"🏠 Predicted House Price: ${prediction:,.2f}"

# ===============================
# 3️⃣ Gradio UI
# ===============================
inputs = [
    gr.Number(label="Living Area (sq ft)"),
    gr.Number(label="Bedrooms"),
    gr.Number(label="Full Bathrooms"),
    gr.Number(label="Half Bathrooms"),
    gr.Number(label="Garage Capacity (cars)"),
    gr.Number(label="Year Built"),
    gr.Number(label="Lot Area (sq ft)"),
    gr.Slider(1, 10, step=1, label="Overall Quality Rating"),
    gr.Slider(1, 10, step=1, label="Overall Condition Rating"),
    gr.Number(label="Basement Area (sq ft)"),
    gr.Number(label="Garage Area (sq ft)"),
    gr.Number(label="First Floor Area (sq ft)"),
    gr.Number(label="Second Floor Area (sq ft)"),
    gr.Number(label="Number of Fireplaces")
]

output = gr.Textbox(label="Predicted Price", interactive=False)

app = gr.Interface(
    fn=predict_house_price,
    inputs=inputs,
    outputs=output,
    title="🏡 House Price Prediction",
    description="Fill in the house details to predict its price."
)

# ===============================
# 4️⃣ Run the app
# ===============================
if __name__ == "__main__":
    app.launch()
