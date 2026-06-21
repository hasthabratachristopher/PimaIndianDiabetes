# Prediksi Diabetes — Perbandingan Model Klasifikasi Machine Learning

A machine learning classification project that predicts the likelihood of diabetes based on clinical features, comparing the performance of five classification algorithms: **Logistic Regression, Decision Tree, Support Vector Machine (SVM), K-Nearest Neighbors (KNN), and Random Forest**.

## Dataset

- **Source**: `Diabetes.csv` (Pima Indians Diabetes dataset)
- **Target variable**: `Outcome` (0 = no diabetes, 1 = diabetes)
- **Features**: includes clinical measurements such as `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `Age`

## Workflow

1. **Exploratory Data Analysis (EDA)**:
   - Inspect data structure and distributions
   - Visualize feature relationships with pair plots colored by `Outcome`
2. **Preprocessing**:
   - **Outlier handling**:
     - `Insulin`: capped (winsorized) using the Interquartile Range (IQR) method — extreme values above the upper bound are replaced with the upper bound rather than removed
     - Other features: outliers identified and removed using **Local Outlier Factor (LOF)** with 10 nearest neighbors
   - Boxplot visualization of outliers per feature before/after treatment
3. **Feature Engineering**:
   - Encoding of categorical/derived features
   - **RobustScaler** applied to normalize feature scales while reducing the influence of remaining outliers
4. **Modeling**:
   - Train/test split
   - Models trained: Logistic Regression, Decision Tree, SVM, KNN, Random Forest
   - Evaluation via accuracy, precision, recall, F1-score, and confusion matrix
5. **Model Evaluation & Comparison**:
   - ROC curves and AUC scores plotted for all models
   - Bar chart comparing accuracy and AUC across models

## Results

| Model | Accuracy | ROC AUC |
|---|---|---|
| Logistic Regression | ~86.84% | ~87% |
| Decision Tree | ~90.13% | ~86% |
| Support Vector Machine | ~88.82% | ~88% |
| K-Nearest Neighbors | ~87.5% | ~86% |
| **Random Forest** | **~90.79%** | **~89%** |

**Random Forest** achieved the best overall performance (highest accuracy and AUC), followed by Decision Tree and SVM.

## Recommendations

- **Random Forest** and **SVM** are the strongest candidates for deployment based on accuracy and AUC.
- **Logistic Regression** remains a solid, simpler alternative with comparable performance.
- Further improvements could include:
  - Hyperparameter tuning (Grid Search / Random Search)
  - Cross-validation (e.g. stratified k-fold) to ensure robustness and avoid overfitting
  - Feature selection or dimensionality reduction (e.g. PCA)
  - Training with additional data to improve generalization

## Tools & Libraries

- `pandas`, `numpy`
- `seaborn`, `matplotlib`
- `statsmodels`
- `scikit-learn` (`LogisticRegression`, `KNeighborsClassifier`, `DecisionTreeClassifier`, `SVC`, `RandomForestClassifier`, `LocalOutlierFactor`, `StandardScaler`/`RobustScaler`, `train_test_split`, `GridSearchCV`, metrics)

## How to Run

```bash
pip install pandas numpy seaborn matplotlib statsmodels scikit-learn
```

1. Place `Diabetes.csv` in the working directory.
2. Run the notebook sequentially in Jupyter or Google Colab.
3. Outputs include EDA plots, outlier-treatment boxplots, classification reports, ROC curves (saved as `PE_diabetes.jpeg`), and a model comparison bar chart.
