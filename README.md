# Conversion in Digital Marketing: EDA and Predictive Modeling

This project focuses on analyzing and predicting conversion rates in digital marketing using various machine learning models and exploratory data analysis (EDA) techniques. It demonstrates how data preprocessing, feature engineering, and model tuning can be applied to real-world marketing data for better predictions and insights.

## Project Overview

### Sections Covered:

1. **Libraries Used:**
   - Data Processing: `pandas`, `numpy`
   - Data Visualization: `seaborn`, `matplotlib`
   - Machine Learning: `sklearn` (e.g., `DecisionTrees`, `RandomForest`, `ExtraTrees`, `GradientBoosting`, `HistGradientBoosting`), `XGBoost`, `LightGBM`, `CatBoost`
   - Model Evaluation: `f1 score`, `confusion matrix`, `precision-recall curve` from `sklearn metrics`
   - Advanced Techniques: `pps score` for predictive power, `SMOTE` for oversampling, `RandomUnderSampler` for undersampling, `TargetEncoder` for categorical encoding

2. **Data Loading:**
   - The data was imported and preprocessed for analysis and modeling.

3. **Exploratory Data Analysis (EDA):**
   - **Skimpy Summary**: Basic statistical insights (mean, standard deviation, quantiles) using `skimpy`.
   - **Unique Values**: Identification of unique values in columns for deeper analysis.
   - **Duplicate Rows Check**: Ensured there were no duplicate rows (0 duplicates found).
   - **Category Classification**: Data classified into discrete, continuous, and categorical variables.
   - **Visualizations**:
     - **Rugplots**: Analyzed the distribution of categories, including mean, quantiles, skewness, and customer behavior insights.
     - **Pie Charts**: Explored data imbalances, especially in conversions (only 12.3% did not convert).
     - **Correlation Matrix**: Explored weak correlations between variables (Pearson & Spearman).
     - **Predictive Power Score (PPS)**: Found very low predictive power (0.00) among features.

4. **Data Preprocessing:**
   - Dropped unnecessary columns (e.g., `customerID`).
   - Applied **Min-Max Scaling** on numerical features.
   - Converted categorical features using **Target Encoding**.
   - Handled class imbalance using **SMOTE** (oversampling) and **undersampling**.
   - Models were trained on three different datasets: the original, oversampled, and undersampled.

5. **Model Training & Predictions:**
   - **Original Dataset**:
     - CatBoost: `0.9642` F1 score
     - Grid Search CV optimization: `0.9684` F1 score with best parameters: `depth: 4`, `iterations: 1000`, `l2_leaf_reg: 5`, `learning_rate: 0.05`
   - **Oversampled Dataset (SMOTE)**:
     - CatBoost: `0.9658` F1 score
     - Grid Search CV optimization: `0.9663` F1 score with best parameters: `depth: 8`, `iterations: 1000`, `l2_leaf_reg: 3`, `learning_rate: 0.05`
   - **Undersampled Dataset**:
     - CatBoost: `0.9463` F1 score (dropped as it performed worst).

6. **Model Evaluation & Metrics:**
   - Evaluated models using **Confusion Matrix** and **Precision-Recall Curves** to assess performance.

## Key Takeaways:
- **SMOTE** and **undersampling** techniques helped tackle class imbalance effectively.
- **CatBoost** and **Grid Search CV** tuning yielded the best results on both the original and oversampled datasets.
- The model can be further improved with additional feature engineering or hyperparameter tuning.

---

Feel free to clone this repo and explore the notebook to see how predictive modeling can be applied to optimize conversion rates in digital marketing!
