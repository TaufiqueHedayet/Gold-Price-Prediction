# Comparative Analysis on Predicting Price Hike with Sources Using Different Machine Learning Algorithms

This repository contains the core Machine Learning and Deep Learning time-series prediction pipelines developed for financial forecasting (specifically gold price fluctuation). This work was presented virtually at the **International Conference on Information and Communication Technology (ICICT 2025), London, UK**, and published with **Springer Nature**.

## 🚀 Key Research Features & Contributions
- **Multi-Model Comparative Framework:** Implemented and compared five advanced machine learning and deep learning models: Support Vector Regression (SVR), AdaBoost, Gradient Boosted Regression Trees (GBRT), LSTM, and Bidirectional LSTM (BLSTM) to determine the highest predictive precision.
- **4-Layer Methodology:** Formulated a structured four-phase engineering pipeline: Data Pre-processing (handling missing values, MinMaxScaler, and StandardScaler), Data Serving, Machine Learning Training/Testing, and Finalization (Result Modeling/Analysis).
- **Model Serialization Layer:** Serialized high-dimensional network architectures into functional execution objects, mapping neural network layers to `.h5` formats and classic regression structures to `.pkl` formats for stable inference.
- **Robust Feature Engineering:** Processed over 3,700 dynamic historical data points (spanning 2010 to 2024) sourced from *Investing.com*, handling multi-dimensional parameters including Open, High, Low, Volume (K/M scaling), and Change%.
- **Live Google Colab Execution:** Fully configured notebook ready to run directly in Google Colab with sequential model mapping layers for quick verification and evaluation.

## 📊 Models & Performance Benchmark
The sequential data structures are mapped over a 60-day historical window across a robust 100-epoch training framework to evaluate predictive accuracy.

| Model Architecture | Mean Squared Error (MSE) | R² Score | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Bidirectional LSTM (BLSTM)** | **349.42** | **0.9983** | **99.79%** | **100%** | **99.59%** | **0.9979** |
| Gradient Boosting (GBRT) | 0.15 | 0.9966 | 99.04% | 99.45% | 98.63% | 0.9904 |
| AdaBoost Regressor | 323.51 | 0.9964 | 99.18% | 99.72% | 98.63% | 0.9918 |
| Long Short-Term Memory (LSTM) | 401.83 | 0.9955 | 97.81% | 97.54% | 98.08% | 0.9781 |
| Support Vector Regression (SVR) | 3187.94 | 0.9667 | 92.55% | 0.9253 | 0.9200 |

*Note: The Deep BLSTM model achieved the highest baseline performance across all time-series evaluations with a 99.79% accuracy rate.*

---

## 💾 Model Artifacts & Serialization Pipeline
To maintain modularity and operational execution without retraining, the framework utilizes dual binary protocols:
1. **`.h5` Structure (Hierarchical Data Format v5):** Retains complete internal tensor parameters, layer weights, sequential connections, and compilation states for **LSTM** and **Bidirectional LSTM (BLSTM)** neural structures.
2. **`.pkl` Structure (Pickle Binary Stream):** Serializes mathematical optimization boundaries, parameters, and ensemble thresholds for classic regression models (**SVR, AdaBoost, GBRT**) alongside feature scaling states (**MinMaxScaler / StandardScaler**).

---

## 🛠️ Tech Stack & Production Tools
- **Core Language:** Python 3.10+
- **Deep Learning Framework:** Keras, TensorFlow (`.h5` serialization)
- **Ensemble & Classic ML:** Scikit-Learn / Joblib (`.pkl` serialization)
- **Data Manipulation & Stats:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn

---

## 📂 Repository Structure
```text
├── Taufique_Gold_Forecasting.ipynb   # Core notebook detailing EDA, data engineering, and multi-model research
├── .gitignore                       # Standard data science rules excluding heavy weight footprints (.pkl/.h5)
└── README.md                        # Complete project documentation matrix and benchmark index



⚙️ How to Run on Google Colab (Execution Guide)
Follow these step-by-step instructions to run the machine learning notebook and reproduce the evaluation charts instantly:

Download the Taufique_Gold_Forecasting.ipynb file from this repository.

Upload the notebook file to your Google Colab workspace (colab.research.google.com).

Download the historical gold commodities raw dataset from Investing.com spanning 2010 to 2024.

Ensure you place your target financial dataset (Gold Futures Historical Data.csv) in your Google Colab root run path directory exactly as: /content/Gold Futures Historical Data.csv.

Run all the code cells sequentially to process feature scaling, execute deep learning model training, and view performance graphs.

📜 Authors & Citation

Taufique Hedayet, Anup Sen, Mahfuza Akter Jarin, Md. Shohel Rana Shaon, Joybordhan Sarkar, and Sadah Anjum Shanto.
Bangladesh University of Business and Technology (BUBT), Dhaka, Bangladesh.
