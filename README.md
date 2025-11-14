# Stellar_ML: Machine Learning for Celestial Object Classification

![SDSS](https://www.sdss.org/)

## Project Overview

**Stellar_ML** is a comprehensive machine learning project that classifies celestial objects (Stars, Galaxies, and Quasars) from the Sloan Digital Sky Survey (SDSS) DR17 dataset. This project demonstrates the complete pipeline from data preprocessing to model interpretation, achieving **98% classification accuracy** with robust astrophysical validation.

## Objectives

- **Classify** celestial objects into three categories: Stars, Galaxies, and Quasars
- **Compare** multiple machine learning algorithms for optimal performance
- **Interpret** model decisions using SHAP analysis for astrophysical validation
- **Validate** that model predictions align with established astronomical principles
- **Address** class imbalance and evaluate clustering feasibility

## Dataset

**Source:** Sloan Digital Sky Survey (SDSS) DR17 - Final release of SDSS-IV phase  
**Samples:** 100,000 spectroscopic observations  
**Features:** 17 numerical and categorical attributes  
**Classes:** GALAXY, QSO (quasar), STAR

### Key Features
- **Photometric**: `u, g, r, i, z` magnitudes (ultraviolet to infrared)
- **Spectroscopic**: `redshift` - primary discriminator
- **Positional**: `alpha` (RA), `delta` (Dec) coordinates
- **Derived**: Color indices (`u_g`, `g_r`, `r_i`, `i_z`)

## Methodology

### Data Preprocessing
- **Feature Engineering**: Created color indices from magnitude bands
- **Class Balancing**: Applied **SMOTE** to address imbalance
  - Original: GALAXY(59,445), QSO(18,961), STAR(21,594)
  - After SMOTE: All classes balanced to 59,445 samples

### Model Development

#### Basic Models
1. **Gaussian Naive-Bayes** - Baseline probabilistic model
2. **Multiclass Logistic Regression** - Linear classification
3. **Scaled Multiclass Logistic Regression** - With feature standardization
4. **K-NN Classifier** - Instance-based learning

#### Advanced Models
1. **Multiclass SVM** - With hyperparameter tuning
2. **Decision Tree** - Interpretable tree-based model
3. **Random Forest** - Ensemble of decision trees with optimal parameters

#### Boosting Algorithms
1. **AdaBoost** - Sequential learning with decision stumps
   - Accuracy: 88.79%
2. **XGBoost** - Optimized gradient boosting
   - **Best Performance: 98% accuracy**
   - Optimal parameters found through rigorous tuning

### Comprehensive Evaluation
For each model, we performed:
- **Hyperparameter Optimization** - Grid search for best parameters
- **ROC-AUC Analysis** - Multi-class ROC curves
- **Classification Reports** - Precision, recall, F1-scores
- **Confusion Matrices** - Error analysis
- **Cross-Validation** - Robust performance estimation

## Key Findings

### Best Performing Model
**XGBoost Classifier** achieved outstanding results:
- **Accuracy: 98%**
- Excellent performance across all classes
- Minimal misclassifications

### Clustering Analysis
- **K-Means Clustering** attempted but achieved only **40% accuracy**
- **Spherical Visualization** showed significant overlap in feature space
- Confirmed that **clustering is not feasible** for this classification task due to overlapping characteristics

### Model Interpretation with SHAP
**SHAP analysis revealed:**
- **Redshift** is the most influential feature (75%+ contribution)
- **Color indices** (`g_r`, `u_g`) provide secondary discrimination
- Positional features (`alpha`, `delta`) have minimal impact
- Model decisions align perfectly with astrophysical principles

### Redshift Distribution Validation
- **Negligible bias** between predicted and true redshift distributions
- Perfect alignment for **STAR** class (z ≈ 0)
- Minimal deviation for **GALAXY** and **QSO** classes
- Confirms model preserves astrophysical consistency

### Dependencies
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost
- shap, imbalanced-learn
- plotly (for 3D visualizations)


## Scientific Validation

This project goes beyond typical ML applications by:
1. **Astrophysical Consistency**: Model decisions align with known celestial object properties
2. **Redshift Validation**: Predicted distributions match true astrophysical measurements
3. **Feature Importance**: SHAP analysis confirms physically meaningful feature rankings
4. **Bias Analysis**: Demonstrated minimal systematic errors in predictions

## References

1. [SDSS DR17 Official Website](https://www.sdss.org/dr17/)
2. [The Seventeenth Data Release of SDSS](https://arxiv.org/abs/2112.02026)
3. SHAP: Lundberg, S. M., & Lee, S. I. (2017)

---
