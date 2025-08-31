# 🎮 Predicting League of Legends Match Outcomes with Logistic Regression (PyTorch)

##📌 Overview
This project uses **logistic regression in PyTorch** to predict whether a team will **win or lose** a League of Legends match based on match statistics.  
It demonstrates the full machine learning pipeline: data preprocessing, training, evaluation, hyperparameter tuning, and feature importance analysis.

—

## 📂 Project Structure
league-lr/
├── data/
│   └── league_of_legends_data_large.csv   (or link in README)
├── notebooks/
│   └── logistic_regression_lol.ipynb
├── src/
│   └── logistic_regression.py
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
├── README.md
└── requirements.txt

—

## ⚙️ Installation
Clone the repo and install dependencies:

```bash
git clone https://github.com/your-username/league-lr.git
cd league-lr
pip install -r requirements.txt

—

## Dependencies:
torch
pandas
scikit-learn
matplotlib
seaborn

—

## Usage
python src/logistic_regression.py

This will:

Train a logistic regression model on the dataset
1. Evaluate it on test data
2. Save results in the results/ folder:
	Confusion matrix
	ROC curve
	Feature importance chart
3. Save model weights (logistic_regression_model.pth)

—

## 📊 Results
Example outputs (on test set):

	Accuracy: ~XX% (depends on data split and hyperparameters)
	Confusion matrix:
	ROC curve:
	Feature importance (top features driving match outcomes):

—

## 🔧 Hyperparameter Tuning
The script also experiments with different learning rates and reports the best test accuracy.

—

## 📖 Learning Goals
1. Apply logistic regression with PyTorch
2. Practice binary classification (win/loss prediction)
3. Use evaluation metrics (classification report, confusion matrix, ROC curve)
4. Explore feature importance in a game dataset
5. Showcase a full ML pipeline in a clean GitHub project

—

## 🙌 Acknowledgments
Dataset source: (https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/rk7VDaPjMp1h5VXS-cUyMg/league-of-legends-data-large.csv)
IBM