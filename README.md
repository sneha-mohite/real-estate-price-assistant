# 🏡 Real Estate Price Assistant

A hybrid machine learning and large language model (LLM) system for real estate price prediction from natural language queries. The assistant uses an XGBoost regression model, and Google's Gemini 2.5 Flash API for natural language parsing and explanation generation, wrapped in a Gradio-based UI.

---

## 🔧 Features

- 💬 Query properties in plain English (e.g., "3 bed, 2 bath in Austin, 1800 sqft").
- 🧠 Intelligent parsing of descriptions using Google's Gemini model.
- 📊 Real-time price predictions using a trained XGBoost model.
- 🏙️ Market insights and explanations from local data context.
- 🖥️ User-friendly Gradio web interface.

---

## 🚀 Getting Started

1. **Clone the repository**
   ```
   git clone https://github.com/yourusername/real-estate-price-assistant.git
   cd real-estate-price-assistant
   ```

2. **Prerequisites**

   Google API Key for Gemini 2.5 Flash

   - Sign up and get access to [GOOGLE API KEY](https://aistudio.google.com/apikey).
   - Store your API key in a `.env` file:
   ```
   GOOGLE_API_KEY=xxxx
   ```

2. **Python & Dependencies**

   - Python 3.9+ is recommended.
   - Create and activate a virtual environment:

   ```
   python -m venv .venv
   source .venv/bin/activate       # On Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```
3. **Model Training**
   
   If the trained model is missing or fails to load, you can retrain it:
   ```
   python src/train_model.py
   ```
   Note: The model is trained using GPU. If you do not have a GPU, modify [line 27 of train_model.py](src/train_model.py) to use CPU.

4. **Using the assistant**
   1. 🧪 [Jupyter Notebook](src/chatbot_test_notebook.ipynb)
   2. 🖼️ Gradio Web Interface
    
      ```
      python src/chatbot_app.py
      ```

The model is trained on the [USA Real Estate Dataset](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset) from Kaggle.