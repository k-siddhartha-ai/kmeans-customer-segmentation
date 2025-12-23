# =========================================================
# Customer Segmentation using K-Means (Streamlit App)
# =========================================================

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation | K-Means",
    layout="centered"
)

st.title("🛒 Customer Segmentation using K-Means")
st.write("Unsupervised Learning | Elbow Method | Silhouette Score")

# ---------------------------------------------------------
# Sidebar Controls
# ---------------------------------------------------------
st.sidebar.header("Controls")

num_customers = st.sidebar.slider(
    "Number of customers",
    min_value=100,
    max_value=1000,
    value=300,
    step=50
)

k_clusters = st.sidebar.slider(
    "Number of clusters (K)",
    min_value=2,
    max_value=10,
    value=4
)

# ---------------------------------------------------------
# Generate Synthetic Customer Data
# ---------------------------------------------------------
np.random.seed(42)

ages = np.random.normal(35, 10, num_customers).astype(int)
annual_income = np.random.normal(50000, 10000, num_customers).astype(int)
spending_score = np.random.uniform(1, 100, num_customers).astype(int)

df = pd.DataFrame({
    "Age": ages,
    "Annual Income": annual_income,
    "Spending Score": spending_score
})

st.subheader("📊 Sample Customer Data")
st.dataframe(df.head())

# ---------------------------------------------------------
# Feature Scaling
# ---------------------------------------------------------
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# ---------------------------------------------------------
# Elbow Method (WCSS)
# ---------------------------------------------------------
st.subheader("📉 Elbow Method")

wcss = []
k_range = range(1, 11)

for k in k_range:
    model = KMeans(n_clusters=k, n_init=10, random_state=42)
    model.fit(scaled_data)
    wcss.append(model.inertia_)

fig1, ax1 = plt.subplots()
ax1.plot(k_range, wcss, marker="o", linestyle="--")
ax1.set_xlabel("Number of clusters (K)")
ax1.set_ylabel("WCSS")
ax1.set_title("Elbow Method for Optimal K")
ax1.grid(True, linestyle="--", alpha=0.6)

st.pyplot(fig1)

# ---------------------------------------------------------
# K-Means Clustering
# ---------------------------------------------------------
model_kmeans = KMeans(
    n_clusters=k_clusters,
    n_init=10,
    random_state=42
)

labels = model_kmeans.fit_predict(scaled_data)
df["Cluster"] = labels

# ---------------------------------------------------------
# Silhouette Score
# ---------------------------------------------------------
sil_score = silhouette_score(scaled_data, labels)

st.subheader("📐 Silhouette Score")
st.write(f"**Silhouette Score for K={k_clusters}:** `{sil_score:.4f}`")

# ---------------------------------------------------------
# Cluster Visualization
# ---------------------------------------------------------
st.subheader("🧠 Customer Segments Visualization")

fig2, ax2 = plt.subplots()
ax2.scatter(
    df["Annual Income"],
    df["Spending Score"],
    c=df["Cluster"],
    cmap="viridis",
    s=50,
    alpha=0.8
)

ax2.set_xlabel("Annual Income")
ax2.set_ylabel("Spending Score")
ax2.set_title("Customer Segmentation using K-Means")
ax2.grid(True, linestyle="--", alpha=0.6)

st.pyplot(fig2)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown("Built with **Streamlit** and **Scikit-learn**")
