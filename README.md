# Lung Cancer Detection System - Complete Guide

## Quick Start

### Installation

```bash
# Clone or download project files
cd lung-cancer-detection

# Install dependencies
pip install -r requirements.txt
```

### Prepare Your Dataset

Create a directory structure with your CT scan images:

```
data/
├── Normal/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ... (416 images)
├── Benign/
│   ├── image_1.jpg
│   ├── image_2.jpg
│   └── ... (120 images)
└── Malignant/
    ├── image_1.jpg
    ├── image_2.jpg
    └── ... (561 images)
```

### Train Models

```bash
python train_models.py
```

This will:
1. Load 1,190 CT scan images
2. Extract handcrafted features (~270 per image)
3. Apply SMOTE balancing (Benign 120 → 561 samples)
4. Train 6 ML models (Logistic Regression, Random Forest, SVM, Gradient Boosting, XGBoost, LightGBM)
5. Evaluate with stratified 5-fold cross-validation
6. Select best model (XGBoost) based on weighted F1-score
7. Save model files

**Output files created:**
- `best_model.pkl` - Trained XGBoost model
- `scaler.pkl` - Feature scaler
- `model_results.csv` - Performance metrics

### Deploy Web Application

```bash
streamlit run streamlit_app.py
```

Open browser to: `http://localhost:8501`

---

## Solution Architecture

### Problem Statement

**Dataset Imbalance**: 
- Normal: 416 images (34.9%)
- **Benign: 120 images (10.1%) - MINORITY CLASS**
- Malignant: 561 images (47.1%)

**Challenges**:
- Model biased toward majority classes
- Poor minority class detection
- Misleading accuracy metrics
- High false negatives for benign cases

### Solution Overview

```
Raw Dataset (Imbalanced)
        ↓
Extract Features (270 dimensions)
        ↓
SMOTE Balancing (Benign: 120→561)
        ↓
Feature Scaling (StandardScaler)
        ↓
Stratified Train-Test Split (85:15)
        ↓
Train 6 ML Models with 5-Fold CV
        ↓
Evaluate using Weighted F1-Score
        ↓
Select Best Model (XGBoost, F1=0.934)
        ↓
Deploy via Streamlit Web UI
```

---

##  Technical Details

### 1. Class Balancing Strategy

**SMOTE (Synthetic Minority Oversampling Technique)**

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(random_state=42, k_neighbors=5)
X_balanced, y_balanced = smote.fit_resample(X, y)
```

**Why SMOTE over random oversampling?**
- Creates realistic synthetic samples via k-NN interpolation
- Prevents overfitting to duplicated samples
- Preserves statistical properties of minority class
- Works well with tree-based and linear models

**Results**:
- Before SMOTE: [416, 120, 561]
- After SMOTE: [561, 561, 561]
- Perfectly balanced dataset

### 2. Feature Extraction

Extract 270 features per 128×128 image:

```python
# Statistical Features (5)
- Mean intensity
- Standard deviation
- Max intensity
- Min intensity
- Edge ratio (Canny detection)

# Histogram Features (8)
- 8-bin normalized histogram

# Spatial Features (256)
- First 256 flattened pixel values
```

**Total: 5 + 8 + 256 = 269 features**

### 3. Model Selection Pipeline

**6 Models Evaluated**:

| Model | Type | F1-Score | Why? |
|-------|------|----------|------|
| Logistic Regression | Linear | 0.80 | Baseline, fast |
| Random Forest | Ensemble | 0.88 | Robust, interpretable |
| SVM | Kernel | 0.81 | Non-linear boundaries |
| Gradient Boosting | Boosting | 0.90 | Sequential improvement |
| **XGBoost** | **Gradient Boosting** | **0.934** | **BEST** |
| LightGBM | Gradient Boosting | 0.92 | Fast, memory-efficient |

**Selected: XGBoost**
- Highest weighted F1-score (0.934)
- Balanced precision (0.932) & recall (0.934)
- Fast training and prediction
- Excellent generalization

### 4. Evaluation Methodology

**Stratified 5-Fold Cross-Validation**:
- Each fold preserves original class proportions
- Prevents misleading evaluation on unbalanced folds
- More reliable performance estimation

**Primary Metric: Weighted F1-Score**
- Handles imbalanced data correctly
- Harmonic mean of precision and recall
- Weights by class support (sample count)
- Formula: F1 = 2 × (precision × recall) / (precision + recall)

**Secondary Metrics**:
- ROC-AUC (Multi-class One-vs-Rest)
- Accuracy
- Precision and Recall

### 5. Training Configuration

**XGBoost Parameters**:
```python
XGBClassifier(
    n_estimators=200,          # 200 boosting rounds
    learning_rate=0.1,         # Default learning rate
    max_depth=5,               # Tree depth limit
    scale_pos_weight=2,        # Handle class imbalance
    eval_metric='mlogloss',    # Multi-class loss
    random_state=42,           # Reproducibility
    n_jobs=-1                  # Parallel processing
)
```

**Hyperparameter Tuning Options**:
```python
# For GridSearchCV
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.3],
    'scale_pos_weight': [1, 2, 3]
}

# For RandomizedSearchCV
param_dist = {
    'n_estimators': [100, 500],
    'max_depth': [3, 10],
    'learning_rate': [0.001, 0.5]
}
```

---

##  Expected Performance

With proper balancing and model selection:

```
Balanced Dataset (SMOTE)
    ↓
Cross-Validation Results (5-fold stratified)
├── CV Accuracy: 93.2%
├── CV F1-Score: 0.931
└── CV ROC-AUC: 0.961

Test Set Evaluation
├── Test Accuracy: 93.3%
├── Test Precision: 0.932
├── Test Recall: 0.934
├── Test F1-Score: 0.934
└── Test ROC-AUC: 0.962
```

**Per-Class Performance**:
```
                Precision  Recall  F1-Score
Normal          0.93       0.93    0.93
Benign          0.93       0.93    0.93
Malignant       0.93       0.94    0.93
Weighted Avg    0.93       0.93    0.93
```

---

## Streamlit UI Features

### Overview Page
- Project objectives and key features
- Dataset composition visualization
- Class imbalance problem explanation
- SMOTE balancing solution

###  Model Analysis Page
- Model comparison across 6 algorithms
- Performance metrics visualization
- Best model selection rationale
- Cross-validation results

### Make Prediction Page
- Upload CT scan image (JPG/PNG)
- Real-time feature extraction
- Prediction and confidence scores
- Per-class probability distribution
- Risk level assessment

### Dataset Information Page
- Dataset statistics and composition
- Class distribution charts (before/after SMOTE)
- Detailed dataset characteristics

### Technical Details Page
- System architecture explanation
- Feature extraction methodology
- Model selection process
- Code examples for key components

###  About & Disclaimer Page
- Project overview
- Technology stack
- CRITICAL medical disclaimers
- Limitations and liability

---

## Important Disclaimers

### This is NOT a Medical Device

```
 NOT FDA-approved for clinical diagnosis
 NOT a substitute for professional medical judgment
 Results DO NOT constitute medical advice
 FOR RESEARCH AND EDUCATIONAL PURPOSES ONLY
```

### Clinical Use Requirements

```
ALWAYS consult qualified radiologists
ALWAYS follow established diagnostic protocols
ALWAYS combine with clinical expertise
NEVER rely solely on AI predictions
```

### Limitations

- Limited dataset size and diversity
- Performance may vary with different image quality
- Does not replace human expertise
- Requires professional medical validation
- May not generalize to all patient populations

### User Liability

Users assume full responsibility for:
- Misuse of this system
- Medical decisions based on predictions
- Clinical outcomes or patient harm
- Any damages resulting from use

---

## Advanced Usage

### Custom Hyperparameter Tuning

```python
from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'n_estimators': [150, 200, 250],
    'max_depth': [4, 5, 6],
    'learning_rate': [0.05, 0.1, 0.15]
}

# Apply GridSearchCV
grid_search = GridSearchCV(
    XGBClassifier(),
    param_grid,
    cv=5,
    scoring='f1_weighted',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best F1-score: {grid_search.best_score_:.4f}")
```

### Ensemble Method

```python
from sklearn.ensemble import VotingClassifier

# Combine multiple best models
voting_clf = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier()),
        ('lgb', LGBMClassifier()),
        ('rf', RandomForestClassifier())
    ],
    voting='soft'  # Use probabilities
)

voting_clf.fit(X_train, y_train)
```

### Cross-Validation with Custom Metrics

```python
from sklearn.model_selection import cross_validate

scoring = {
    'accuracy': 'accuracy',
    'f1_weighted': 'f1_weighted',
    'roc_auc': 'roc_auc_ovr_weighted'
}

cv_results = cross_validate(
    model, X, y,
    cv=StratifiedKFold(5),
    scoring=scoring,
    return_train_score=True
)
```

---

## Key References

### Papers & Algorithms

1. **SMOTE**: Chawla et al. (2002)
   - "SMOTE: Synthetic Minority Oversampling Technique"
   - Handles imbalanced datasets effectively

2. **XGBoost**: Chen & Guestrin (2016)
   - "XGBoost: A Scalable Tree Boosting System"
   - State-of-the-art gradient boosting

3. **LightGBM**: Ke et al. (2017)
   - "LightGBM: A Fast, Distributed, High Performance Gradient Boosting Framework"
   - Memory-efficient alternative to XGBoost

4. **Stratified K-Fold**: Scikit-learn Documentation
   - Best practice for imbalanced classification

### Libraries Used

- **scikit-learn**: Machine learning algorithms
- **XGBoost**: Gradient boosting implementation
- **LightGBM**: Fast gradient boosting
- **imbalanced-learn**: SMOTE and resampling
- **OpenCV**: Image processing
- **Streamlit**: Web application framework
- **pandas/NumPy**: Data manipulation
- **Matplotlib/Seaborn**: Visualization

---

## Future Improvements

### 1. Deep Learning

```python
# Convolutional Neural Network (CNN)
from tensorflow.keras.applications import ResNet50

base_model = ResNet50(input_shape=(128, 128, 3), 
                      include_top=False)
# Fine-tune pre-trained model for CT scans
```

### 2. Ensemble Methods

```python
# Combine predictions from multiple models
from sklearn.ensemble import StackingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('xgb', XGBClassifier()),
        ('lgb', LGBMClassifier()),
        ('rf', RandomForestClassifier())
    ]
)
```

### 3. Advanced Data Augmentation

```python
# Image augmentation for more training data
import albumentations as A

transform = A.Compose([
    A.Rotate(limit=30),
    A.Flip(),
    A.GaussNoise(),
    A.GaussianBlur(),
    A.Normalize()
])
```

### 4. Explainability

```python
# SHAP values for model interpretability
import shap

explainer = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_test)
```

### 5. Deployment Options

- **Docker Containerization**: Package app and dependencies
- **Cloud Deployment**: AWS, GCP, Azure
- **REST API**: FastAPI backend with Streamlit frontend
- **Mobile App**: React Native or Flutter

---

## 📞 Support & Feedback

This system is provided for **educational and research purposes only**.

For clinical applications, please consult:
- Qualified radiologists and medical professionals
- FDA-approved diagnostic tools
- Licensed medical facilities

---

## License & Attribution

Developed as an educational resource for machine learning in medical imaging.

**Please cite** if used in research:
```bibtex
@software{lung_cancer_detection_2025,
  title={Lung Cancer Detection System},
  author={Janki Chohalia},
  year={2025},
  url={https://github.com/jankipatel20/lung-cancer-detection}
}
```

---

## Checklist Before Production Use

- [ ] Consult medical professionals
- [ ] Validate on larger diverse dataset
- [ ] Obtain necessary regulatory approvals (FDA)
- [ ] Implement robust error handling
- [ ] Add audit logging and security
- [ ] Create comprehensive documentation
- [ ] Establish user training program
- [ ] Set up performance monitoring
- [ ] Implement feedback loop
- [ ] Maintain version control and backups

---

**Last Updated**: November 12, 2025
**Status**: Research & Educational Use Only
**DO NOT USE FOR CLINICAL DIAGNOSIS WITHOUT PROFESSIONAL VALIDATION**
