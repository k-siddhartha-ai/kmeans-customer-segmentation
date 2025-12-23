# 🛒 Customer Segmentation using K-Means Clustering

An interactive **customer segmentation web app** built using **K-Means clustering**.  
This project demonstrates **unsupervised machine learning**, using the **Elbow Method** and **Silhouette Score** to determine the optimal number of clusters.

🚀 **Live Demo (Hugging Face App):**  
https://huggingface.co/spaces/Siddhartha001/kmeans-customer-segmentation


## 📌 Project Overview

Customer segmentation helps businesses understand different customer groups based on their behavior and attributes.  
This application groups customers using:

- **Age**
- **Annual Income**
- **Spending Score**

The app allows users to interactively:
- Choose the number of customers
- Select the number of clusters (K)
- Visualize clusters in real time
- Evaluate clustering quality using Silhouette Score


## 🧠 Machine Learning Techniques Used

- **K-Means Clustering**
- **Elbow Method (WCSS)**
- **Silhouette Score**
- **Feature Scaling (StandardScaler)**


## 🛠️ Tech Stack

- **Python**
- **Streamlit**
- **Scikit-learn**
- **NumPy**
- **Pandas**
- **Matplotlib**
- **Hugging Face Spaces**


## ⚙️ How the App Works

1. Synthetic customer data is generated
2. Data is scaled using StandardScaler
3. Elbow Method visualizes optimal K
4. K-Means clusters the customers
5. Silhouette Score evaluates cluster quality
6. Results are displayed in an interactive plot

---

## ▶️ Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py

```

📂 Project Structure
kmeans-customer-segmentation/
│
├── app.py              # Streamlit application
├── requirements.txt    # Dependencies
└── README.md           # Project documentation

🌐 Deployment

Deployed using Hugging Face Spaces

Streamlit SDK

Automatic builds on file updates

Live App 👉
https://huggingface.co/spaces/Siddhartha001/kmeans-customer-segmentation

📈 Use Cases

Marketing strategy optimization

Targeted promotions

Customer behavior analysis

Business intelligence dashboards

👤 Author

Karne Siddhartha

GitHub: https://github.com/
<your-github-username>

Hugging Face: https://huggingface.co/Siddhartha001

⭐ If you like this project

Give it a ⭐ on GitHub and a ❤️ on Hugging Face!
