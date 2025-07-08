import gradio as gr
import pandas as pd
import joblib
import numpy as np
from utils import parse_query_with_llm, prepare_features, generate_explanation

model_pipeline = joblib.load("models/real_estate_model_xgboost_gpu_pipeline.joblib")
context_df = pd.read_csv("data/usa-real-estate-dataset.csv")
context_df['price_per_sqft'] = context_df['price'] / context_df['house_size']

def predict_price(user_query: str):
    parsed = parse_query_with_llm(user_query)
    if not parsed:
        return "❌ Could not parse the query."

    features_df = prepare_features(parsed, context_df)
    log_price = model_pipeline.predict(features_df)[0]
    price = round(np.expm1(log_price), -3)
    explanation = generate_explanation(user_query, price, parsed, context_df)
    return f"💰 **Predicted Price:** ${price:,.0f}\n\n🧠 **Explanation:**\n{explanation}"

with gr.Blocks(css=".gr-container {justify-content: center; align-items: center; text-align: center;}") as demo:
    with gr.Column(scale=1, min_width=500):
        gr.Markdown("## 🏠 Real Estate Price Assistant")
        gr.Markdown("Ask about a property and get an estimated price with market context and explanation.")

        input_box = gr.Textbox(placeholder="e.g. 3 bed, 2 bath house in Austin, TX, 1800 sqft, built 2015", label="Property Description")
        output_text = gr.Markdown()

        submit_button = gr.Button("Estimate Price")
        submit_button.click(fn=predict_price, inputs=input_box, outputs=output_text)

# Launch the Gradio application
if __name__ == "__main__":
    demo.launch()