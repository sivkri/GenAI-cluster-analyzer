# Interactive Clustering with GenAI

An interactive **Streamlit-based application** for performing **traditional and deep learning-based clustering**, enhanced with **explainable AI (XAI)** and **Generative AI (GenAI)**-powered insights.

---

## Features

### Clustering Techniques
- **Centroid-based**: KMeans
- **Density-based**: DBSCAN, HDBSCAN
- **Connectivity-based**: Agglomerative Hierarchical Clustering (Ward, Complete, Single, Average)
- **Distribution-based**: Gaussian Mixture Models (GMM)
- **Grid-based**: *(Optional extension)*
- **Self-Organizing Maps (SOM)**
- **Deep Learning-based**: Autoencoder + KMeans

### Evaluation Metrics
- Silhouette Score
- Davies-Bouldin Index
- Calinski-Harabasz Index
- Adjusted Rand Index (ARI)
- Normalized Mutual Information (NMI)

### Dimensionality Reduction & Visualization
- PCA, t-SNE, UMAP
- Visualize cluster separation, outliers, and centroids
- Supports both **Matplotlib** and **Plotly** for interactive visuals

### GenAI Features
- **Cluster Summary Generator**: Uses Hugging Face API (`google/flan-t5-base`) to summarize cluster characteristics
- Accepts Hugging Face API key via input prompt
- Automatically generates human-readable insights based on cluster stats

---

## How to Use

### 1. Clone this repository
```bash
git clone  origin https://github.com/sivkri/GenAI-cluster-analyzer           
cd GenAI-cluster-analyzer
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
````

### 3. Run the Streamlit app
```bash
streamlit run streamlit_bot.py
````

### App Flow
1. Upload your dataset (CSV format)

2. Preview the first 5 rows

3. Select columns to drop

4. Select features for clustering

5. Choose clustering method(s)

6. View evaluation metrics and interactive visualizations

7. Optionally generate a GenAI-powered cluster summary

---

### Hugging Face API

To enable GenAI cluster summarization:

1. Go to Hugging Face Tokens

2. Generate a read access token

3. Paste your API key into the app when prompted
