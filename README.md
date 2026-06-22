# Comparative Analysis on Predicting Price Hike with Sources Using Different Machine Learning Algorithms

This repository contains an end-to-end multi-model time-series prediction pipeline developed to forecast price fluctuations for necessary commodities (benchmarked against gold value matrices). This research was presented virtually at the **International Conference on Information and Communication Technology (ICICT 2025), London, UK**, and published with **Springer Nature**.

## 🔄 Core Project Engineering Pipeline
The project lifecycle is split into two distinct engineering environments to mimic industry deployment logic:
1. **Model Testing & Research Phase:** Built, tested, and optimized five comparative algorithms (SVR, AdaBoost, GBRT, LSTM, and Bidirectional LSTM) within a Google Colab tracking notebook (`Taufique_Gold_Forecasting.ipynb`)[cite: 2].
2. **Production & Deployment Phase:** Serialized the best-performing trained network architecture weights and deployed them inside a lightweight **FastAPI REST API wrapper microservice** to process real-world client data inference inputs[cite: 1].

---

## 📊 Models & Performance Benchmark
The models were trained on over 3,700 structured historical data arrays using a 60-day historical window to mitigate volatility and forecast future trend directions[cite: 2].

| Model Architecture | Mean Squared Error (MSE) | R² Score | Accuracy | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Bidirectional LSTM (BLSTM)** | **349.42** | **0.9983** | **99.79%** | **100%** | **99.59%** |
| Long Short-Term Memory (LSTM) | - | - | - | - | - |
| Gradient Boosted Trees (GBRT) | 0.15 | 0.9966 | 99.04% | 0.9945 | 0.9863 |
| AdaBoost Regressor | 323.51 | 0.9964 | 99.18% | 0.9972 | 0.9863 |
| Support Vector Regression (SVR) | 3187.94 | 0.9667 | 92.55% | 0.9253 | 0.9200 |

*Note: The computational structures leverage both statistical error reduction algorithms and recurrent networks[cite: 2].*

---

## 🛠️ Tech Stack & Production Tools
- **Core Language:** Python 3.10+[cite: 2]
- **Deep Learning Framework:** Keras, TensorFlow[cite: 2]
- **Ensemble & Classic ML Framework:** Scikit-Learn[cite: 2]
- **Microservice Layer Framework:** FastAPI, Uvicorn, Pydantic[cite: 1]
- **Data Engineering Pipeline:** Pandas, NumPy[cite: 2]

---

## 📂 Repository Structure
```text
├── Taufique_Gold_Forecasting.ipynb # Core notebook detailing data preprocessing, EDA, and model testing phase
├── app/
│   ├── main.py                     # FastAPI server script containing model loading and routing logic
│   └── trained_models/             # Production weights storage directory (.h5 neural network weights)
├── requirements.txt                # System dependency list for swift package installation
├── .gitignore                      # Git configuration to exclude localized heavy files and binaries
└── README.md                       # Structural project summary and installation index
