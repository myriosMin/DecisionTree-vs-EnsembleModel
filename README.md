# 🧠 Parkinson's Disease Classification with Decision Tree & Stacking Ensemble

A machine learning classification project for identifying Parkinson's Disease based on clinical voice measurements. This project compares a single Decision Tree model with a custom **Stacking Ensemble** combining multiple classifiers for improved generalization.

---

## 🎯 Objective

Build and evaluate two ML models to classify Parkinson's status using patient voice data:

- A basic **Decision Tree Classifier**
- A **Stacked Ensemble Model** (Random Forest, AdaBoost, SVC + Gradient Boosting meta-model)

---

## 📁 Dataset

- `parkinson_disease_assignment.csv` [tweaked by our lecturer]
- Contains biomedical voice measurements from individuals, some with Parkinson’s Disease
- Target: `status` (0 = healthy, 1 = Parkinson’s)

---

## 🧪 Models Developed

### 🌳 Decision Tree Classifier (`t2_233523A_dt`)

- Basic interpretable model
- Tuned with `max_depth`, `min_samples_split`, `min_samples_leaf`
- Evaluated using accuracy, F1-score, confusion matrix, ROC-AUC

### 🤖 Stacking Ensemble Classifier (`t2_233523A_stack`)

- Base models:
  - `RandomForestClassifier`
  - `AdaBoostClassifier`
  - `SupportVectorClassifier`
- Final estimator: `GradientBoostingClassifier`
- Tuned via `GridSearchCV`
- Outperformed DT in generalization and robustness

---

## 🧠 Evaluation Metrics

- Accuracy
- F1-score
- ROC-AUC
- Confusion matrix
- Classification report

---

## 📊 Results Summary

| Model             | Accuracy   | F1-Score   | ROC-AUC    |
| ----------------- | ---------- | ---------- | ---------- |
| Decision Tree     | \~0.85     | \~0.83     | \~0.87     |
| Stacking Ensemble | **\~0.92** | **\~0.91** | **\~0.94** |

> The stacking model consistently outperformed the decision tree across all metrics, especially in handling class imbalance and overfitting.

---

## 🧰 Tech Stack

- Python (Google Colab)
- `sklearn`: model building, tuning, evaluation
- `pandas`, `numpy`, `matplotlib`, `seaborn`: EDA & plotting

---

## 📂 Notebook Structure

```
DecisionTree_vs_EnsembleModels.ipynb
├── Data Preparation
├── Decision Tree Modeling
├── Stacking Ensemble Modeling
├── Model Evaluation
├── Summaries & Comparison
```

---

## 📌 Reflections

- Decision Trees are interpretable but sensitive to overfitting.
- Stack Ensemble provides better generalization at the cost of interpretability and training time.
- Careful hyperparameter tuning (especially for the meta-learner) is crucial.

---

## 👤 Author

[Year 2, Machine Learning Assignment, Diploma in AI & Data Engineering, Nanyang Polytechnic]\
[**Min Phyo Thura**](https://github.com/myriosMin) 

---

Thanks for checking out this project! Feel free to explore the notebook and try optimizing it further 🔍