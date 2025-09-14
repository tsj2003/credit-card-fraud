🚨 Credit Card Fraud Detection using Logistic Regression - 

This project demonstrates how to build a fraud detection model using Logistic Regression on a credit card dataset. It includes a Streamlit web app that allows users to input transaction features and receive instant feedback on whether a transaction is legitimate or fraudulent.

📁 Dataset
The dataset used is the Credit Card Fraud Detection dataset from Kaggle. It contains transactions made by European cardholders in September 2013.

1.Total transactions: 284,807
2.Fraudulent transactions: 492
3.Features are anonymized using PCA (V1 to V28), along with Time and Amount.

✅ Features
1.Balanced training data using undersampling
2.Trained with Logistic Regression
3.Standardized features using StandardScaler
4.Real-time predictions using a Streamlit web app
5.Handles user input with validation
6.Displays model accuracy

📊 Model Performance

Training Accuracy  	~99%
Testing Accuracy  	~98%

💻 How to Run

1. Clone the repository
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection

2. Install dependencies
pip install -r requirements.txt

3. Run the Streamlit app
streamlit run app.py

🧠Usage Instructions
Input all 30 feature values (Time, V1 to V28, Amount) as comma-separated numbers in the input box.

Click Submit to see the prediction:
✅ Legitimate transaction
❌ Fraudulent transaction

Example input:
0, -1.359807, -0.072781, 2.536347, 1.378155, -0.338321, 0.462388, 0.239599, 0.098698, 0.363787, 0.090794, -0.5516, -0.6178, -0.99139, -0.31117, 1.46818, -0.4704, 0.2079, 0.0258, 0.4039, 0.2514, -0.0183, 0.2778, -0.1105, 0.0669, 0.1285, -0.1891, 0.1336, 0


📁 File Structure

credit-card-fraud-detection/
│
├── Credit Card Fraud Detection.ipynb     # Jupyter notebook for training
├── app.py                                # Streamlit app
├── creditcard.csv                        # Dataset (place your dataset here)
├── model.pkl                             # Trained model (optional if pre-saved)
├── scaler.pkl                            # Scaler for feature normalization
├── requirements.txt                      # Dependencies
└── README.md                             # Project documentation


🛠 Technologies Used
Python 🐍
Pandas / NumPy
Scikit-learn
Streamlit
Jupyter Notebook


## 🔗 Live Streamlit App
👉 [Click here to try the model in action](https://anchal-credit-card-fraud-detection-model-tvczvmsf3idfuqtee3w5bd.streamlit.app/)

This release contains:
- Trained logistic regression model
- Streamlit frontend
- Undersampling logic

