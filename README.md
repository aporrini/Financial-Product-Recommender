# 💼 Financial Product Recommender

This project is a Streamlit web application that helps users identify the most suitable financial product based on their profile.

The model uses machine learning techniques to analyze user data (such as age, income, risk propensity, and more) and returns a personalized recommendation.

---

## 🚀 Try the App
###
[![Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://financial-instrument-recommender-fintech-class-2025.streamlit.app/)


[Streamlit Cloud](https://streamlit.io/cloud)

---

## 🧠 How it Works

- Users fill out a short questionnaire
- A trained model runs in the background to predict their ideal financial product
- The app displays a personalized output generated from the model

---

## 🗂️ Repository Structure

```
├── app.py                  # Streamlit app
├── Final_proj.ipynb        # Original Jupyter Notebook
├── Dataset2_Needs.xls      # Dataset on which we trained the ML models
├── requirements.txt        # Python dependencies
├── README.md               # README document
└── models_trained          # Folder with the weights of the main trained models
   └── ...              
```

---

## ⚙️ How to Run Locally

1. Clone the repo:
   ```bash
   git clone https://github.com/aporrini/Financial-Product-Recommender.git
   cd Financial-Product-Recommender
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

---

## 📬 Feedback

If you find a bug or want to suggest improvements, feel free to open an issue or pull request.

---

© 2025 [Alessio Porrini](https://github.com/aporrini)
