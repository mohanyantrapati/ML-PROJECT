import pandas as pd
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder
from streamlit_js_eval import streamlit_js_eval
import requests

# Load the dataset
df = pd.read_csv("Dataset .csv")

# Preprocessing
df = df.dropna(subset=['Cuisines'])  # Remove entries with missing cuisines
df['Cuisines'] = df['Cuisines'].apply(lambda x: x.split(',')[0].strip())  # Simplify cuisines

# Create features for recommendation
features = df[['Cuisines', 'Price range', 'Aggregate rating', 'City']].copy()
le = LabelEncoder()
features['Cuisines'] = le.fit_transform(features['Cuisines'])

# Combine all features into a single string
def combine_features(row):
    return f"{row['Cuisines']} {row['Price range']} {row['Aggregate rating']} {row['City']}"

features_combined = features.apply(combine_features, axis=1)

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(features_combined)

# Streamlit Interface
st.title("🍽️ Intelligent Restaurant Recommender")
st.markdown("Suggesting top-rated restaurants based on your preferences and location.")

# Location Detection
location = streamlit_js_eval(js_expressions="geo", key="get_geo")
user_city = None

if location and 'latitude' in location and 'longitude' in location:
    lat, lon = location['latitude'], location['longitude']
    try:
        response = requests.get(f"https://nominatim.openstreetmap.org/reverse?format=jsonv2&lat={lat}&lon={lon}")
        if response.ok:
            data = response.json()
            user_city = data.get("address", {}).get("city", "")
    except Exception:
        pass

# Manual fallback input
city_input = st.text_input("Detected City (editable):", value=user_city or "", max_chars=50)
cuisine = st.selectbox("Select your preferred cuisine:", df['Cuisines'].unique())
price_range = st.slider("Preferred price range (1 - 4):", 1, 4, 2)
min_rating = st.slider("Minimum rating:", 0.0, 5.0, 3.5, step=0.1)

# Recommendation logic
def recommend_restaurants(city, cuisine, price_range, min_rating, top_n=5):
    try:
        encoded_cuisine = le.transform([cuisine])[0]
    except ValueError:
        return pd.DataFrame([{"Error": f"Cuisine '{cuisine}' not found."}])

    user_feature = f"{encoded_cuisine} {price_range} {min_rating} {city}"
    user_vec = vectorizer.transform([user_feature])
    sim_scores = cosine_similarity(user_vec, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['Restaurant Name', 'Cuisines', 'Price range', 'Aggregate rating', 'City']]

# Show recommendations
if st.button("Recommend"):
    if city_input:
        results = recommend_restaurants(city_input, cuisine, price_range, min_rating)
        st.dataframe(results)
    else:
        st.warning("Please enter or detect a valid city.")
