import json
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
from google import genai
# from google import generativeai as genai


load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
client = genai.Client()

def parse_query_with_llm(user_query: str) -> dict:
    prompt = f"""
        You are a helpful assistant that extracts property features from text.

        Return JSON with only:
        "bedrooms", "bathrooms", "house_size", "city", "state", "built_year", "zip_code", "acre_lot".

        Ignore irrelevant values and only give back the keys if they are present in the user query.
        if a metric is mentioned ignore it. For example house size is 1,800 sqft then just extract 1800.

        Query: "{user_query}"

        JSON:
        """
    try:
        response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
        text = response.text.strip()
        if text.startswith("```json"): text = text[len("```json"):].strip()
        if text.endswith("```"): text = text[:-3].strip()
        return json.loads(text)
    except:
        return {}

def prepare_features(parsed: dict, context_df: pd.DataFrame) -> pd.DataFrame:
    city_from_parsed = parsed.get('city')
    zip_code_from_parsed = parsed.get('zip_code')

    # Impute zip_code
    if zip_code_from_parsed and str(zip_code_from_parsed).strip().lower() != 'nan':
        final_zip_code = str(zip_code_from_parsed).zfill(5)
    elif city_from_parsed and not context_df.empty:
        city_zip_codes = context_df[
            context_df['city'].astype(str).str.contains(str(city_from_parsed), case=False, na=False)
        ]['zip_code'].dropna().astype(str)

        if not city_zip_codes.empty:
            final_zip_code = city_zip_codes.mode().iloc[0]  # use mode (most common zip)
            print(f"Imputed zip code '{final_zip_code}' for city '{city_from_parsed}'.")
        else:
            final_zip_code = '00000'
    else:
        final_zip_code = '00000'

    # --- Construct feature row ---
    row = {
        'bed': float(parsed.get('bedrooms', np.nan)),
        'bath': float(parsed.get('bathrooms', np.nan)),
        'house_size': float(parsed.get('house_size', np.nan)),
        'acre_lot': float(parsed.get('acre_lot', np.nan)),
        'city': city_from_parsed if city_from_parsed else 'Unknown',
        'zip_code': final_zip_code,
        # 'zip_prefix': final_zip_code[:3],
        'brokered_by': parsed.get('state', np.nan),
        'years_since_last_sale': -1,
        'state': parsed.get('state', np.nan),
        'status': parsed.get('state', np.nan),
        'street': parsed.get('street', np.nan),
        
    }

    df = pd.DataFrame([row])

    expected_cols = ['brokered_by', 'status', 'bed', 'bath', 'acre_lot', 'street',
       'city', 'state', 'zip_code', 'house_size', 'prev_sold_date']

    for col in expected_cols:
        if col not in df.columns:
            df[col] = np.nan

    return df[expected_cols]

def generate_explanation(user_query: str, price: float, parsed: dict, context_df: pd.DataFrame) -> str:
      city = parsed.get('city')
      state = parsed.get('state')
      insights = ""

      if not context_df.empty and city and state:
          local = context_df[
              context_df['city'].astype(str).str.contains(city, case=False, na=False) &
              context_df['state'].astype(str).str.contains(state, case=False, na=False)
          ]
          if not local.empty:
              avg_pps = local['price_per_sqft'].mean()
              median_price = local['price'].median()
              median_size = local['house_size'].median()
              insights = (
                  f"- Avg price/sqft in {city}: ${avg_pps:,.0f}/sqft.\n"
                  f"- Median size: {median_size:,.0f} sqft | Median price: ${median_price:,.0f}\n"
              )

      prompt = f"""
          The user asked: "{user_query}"
          Predicted price: ${price:,.0f}

          Property:
          - Bedrooms: {parsed.get("bedrooms")}
          - Bathrooms: {parsed.get("bathrooms")}
          - Size: {parsed.get("house_size_sqft")} sqft
          - Location: {city}, {state}
          - Built: {parsed.get("built_year")}
          - Lot: {parsed.get("acre_lot")} acres

          Market Context:
          {insights if insights else "No local market data available."}

          Give a short, 2-3 sentence explanation of this price and property. Mention 1-2 factors influencing the estimate.
      """
      try:
          response = client.models.generate_content(model="gemini-2.5-flash", contents=prompt)
          return response.text.strip()
      except Exception as e:
          print(f"❌ LLM explanation error: {e}")
          return "No explanation available."
