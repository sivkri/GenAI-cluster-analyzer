import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import plotly.express as px
import requests
import json

st.set_page_config(layout="wide")
st.title("🔍 Clustering App with GenAI Summary")

# Load Data
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("Preview of Uploaded Data")
    st.write(df.head())

    # Ask which columns to drop
    drop_cols = st.multiselect("Select columns to drop", options=df.columns.tolist())
    if drop_cols:
        df.drop(columns=drop_cols, inplace=True)
        st.success(f"Dropped columns: {drop_cols}")
        st.write(df.head())

    # Ask which columns to include in clustering
    cluster_cols = st.multiselect("Select columns to use for clustering", options=df.columns.tolist())
    if cluster_cols:
        df_cluster = df[cluster_cols].copy()

        # Encode categoricals
        for col in df_cluster.select_dtypes(include="object"):
            le = LabelEncoder()
            df_cluster[col] = le.fit_transform(df_cluster[col])

        X_scaled = StandardScaler().fit_transform(df_cluster)

        # KMeans Clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        df_with_labels = df.copy()
        df_with_labels['Cluster'] = kmeans_labels

        # PCA Visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        pca_df = pd.DataFrame(X_pca, columns=['PCA1', 'PCA2'])
        pca_df['Cluster'] = kmeans_labels
        fig = px.scatter(pca_df, x='PCA1', y='PCA2', color=pca_df['Cluster'].astype(str), title="KMeans Clusters (PCA)")
        st.plotly_chart(fig, use_container_width=True)

        # Cluster Summary
        summary_df = df_with_labels.groupby("Cluster")[cluster_cols].mean(numeric_only=True).reset_index()
        st.subheader("📊 Cluster Summary")
        st.dataframe(summary_df)

        # GenAI Cluster Summary
        st.subheader("🧠 Generate Cluster Insights with GenAI")
        api_key = st.text_input("Enter your HuggingFace API key", type="password")
        if api_key:
            prompt = f"Provide a brief insight and interpretation of the following cluster-wise summary:\n\n{summary_df.to_string(index=False)}"
            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": prompt
            }
            response = requests.post(
                "https://api-inference.huggingface.co/models/google/flan-t5-base",
                headers=headers,
                data=json.dumps(payload)
            )
            if response.status_code == 200:
                genai_output = response.json()
                genai_summary = genai_output[0]['generated_text'] if isinstance(genai_output, list) else genai_output.get('generated_text', '')
                st.markdown("### 📝 GenAI Summary")
                st.write(genai_summary)
            else:
                st.error(f"Error generating summary: {response.status_code} - {response.json()}")
