import streamlit as st
import pandas as pd
import joblib


scaler = joblib.load("scaler.pkl")
kmeans = joblib.load("kmeans_model.pkl")

cluster_labels = {
    0: "Introvert",
    1: "Ambivert",
    2: "Extrovert",

}

st.title("People Segmentation using K-Means")
st.markdown("Enter people details to predict their segment.")


talk = st.number_input("talkativeness", min_value=1, max_value=10, value=5)
alone_time = st.number_input("'alone_time_preference", min_value=1, max_value=10, value=5)

# Predict cluster
if st.button("Predict Cluster"):
    new_data = pd.DataFrame([[talk, alone_time]], columns=['talkativeness', 'alone_time_preference'])
    new_scaled = scaler.transform(new_data)
    cluster = kmeans.predict(new_scaled)[0]
    st.success(f"Predicted Cluster: {cluster} - {cluster_labels.get(cluster, 'Unknown')}")
