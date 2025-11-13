import os

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix, 
                            classification_report, roc_curve, auc)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

class LungCancerDataLoader:
    """Load and preprocess CT scan images from directory structure"""
    
    def __init__(self, data_dir='data/', image_size=(128, 128)):
        """
        Initialize data loader for your specific file structure
        
        Args:
            data_dir: Root directory
            image_size: Target image size (width, height)
        """
        self.data_dir = data_dir
        self.image_size = image_size
        self.images = []
        self.labels = []
        
        
        self.possible_class_names = {
            'Normal cases': 0,
            'Benign cases': 1,      
            'Malignant cases': 2
        }
        
        self.class_names = []
        self.class_names_short = []
        self.label_map = {}
        
    def auto_detect_classes(self):
        """Auto-detect which classes are present in the data directory"""
        detected_classes = []
        
        for possible_class in self.possible_class_names.keys():
            class_dir = os.path.join(self.data_dir, possible_class)
            if os.path.exists(class_dir):
                detected_classes.append(possible_class)
        
        if not detected_classes:
            raise ValueError(f" No class directories found in {os.path.abspath(self.data_dir)}")
        
        detected_classes = sorted(detected_classes)
        
        label_counter = 0
        seen_labels = set()
        
        for class_name in detected_classes:
            label = self.possible_class_names[class_name]
            
           
            if label in seen_labels:
                continue
            
            seen_labels.add(label)
            self.class_names.append(class_name)
            short_name = class_name.replace(' cases', '')
            self.class_names_short.append(short_name)
            self.label_map[class_name] = label
        
        print(f"\n AUTO-DETECTED {len(self.class_names)} CLASSES:")
        for i, (full_name, short_name) in enumerate(zip(self.class_names, self.class_names_short)):
            print(f"   Class {i}: {short_name:12} (folder: '{full_name}')")
        
        return self.class_names
        
    def load_images(self, verbose=True):
        """
        Load all images from directory structure
        
        Returns:
            X: Array of images
            y: Array of labels
        """
        
        self.auto_detect_classes()
        
        for class_name in self.class_names:
            class_dir = os.path.join(self.data_dir, class_name)
            
            if not os.path.exists(class_dir):
                print(f" Warning: Directory '{class_dir}' not found")
                continue
            
            image_files = [f for f in os.listdir(class_dir) 
                          if f.lower().endswith(('.jpg', '.png', '.jpeg', '.dcm', '.bmp', '.tiff'))]
            
            if verbose:
                label = self.label_map[class_name]
                short_name = ["Normal", "Benign", "Malignant"][label]
                print(f" Loading {len(image_files)} images from '{class_name}/' ({short_name})...")

            
            for idx, img_name in enumerate(image_files):
                img_path = os.path.join(class_dir, img_name)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    
                    img = cv2.resize(img, self.image_size)
                    
                    self.images.append(img)
                    self.labels.append(self.label_map[class_name])
                    
                except Exception as e:
                    print(f"  Error loading {img_name}: {e}")
        
        images = np.array(self.images)
        labels = np.array(self.labels)
        
        if verbose:
            print(f"\n📊 TOTAL IMAGES LOADED: {len(images)}")
            unique_labels = np.unique(labels)
            for label in unique_labels:
                count = np.sum(labels == label)
                short_name = self.class_names_short[label]
                percentage = (count / len(labels) * 100)
                print(f"   {short_name:12}: {count:4} images ({percentage:5.1f}%)")
        
        return images, labels
    
    def extract_features(self, images, verbose=True):
        """
        Extract handcrafted features from images
        
        Features extracted:
        - Statistical: mean, std, max, min, edge ratio
        - Histogram: 8-bin normalized histogram
        - Spatial: First 256 flattened pixel values
        
        Args:
            images: Array of images
            verbose: Print progress
        
        Returns:
            features: (n_samples, n_features) array
        """
        if verbose:
            print(f"\n🔍 Extracting features from {len(images)} images...")
        
        features_list = []
        
        for idx, img in enumerate(images):
            features = []
            features.append(np.mean(img))              
            features.append(np.std(img))               
            features.append(np.max(img))             
            features.append(np.min(img))          
            edges = cv2.Canny(img, 100, 200)
            edge_ratio = np.sum(edges) / (edges.shape[0] * edges.shape[1])
            features.append(edge_ratio)
            
            
            hist = cv2.calcHist([img], [0], None, [8], [0, 256])
            hist_normalized = hist.flatten() / (np.sum(hist) + 1e-10)
            features.extend(hist_normalized)
            
            
            flat_image = img.flatten()
            spatial_features = flat_image[:256]
            features.extend(spatial_features)
            
            
            combined_features = np.array(features)
            features_list.append(combined_features)
            
            if verbose and (idx + 1) % 300 == 0:
                print(f"   ✅ Processed {idx + 1}/{len(images)} images")
        
        features_array = np.array(features_list)
        
        if verbose:
            print(f"\n✅ Feature extraction complete!")
            print(f"   Shape: {features_array.shape}")
            print(f"   Total features per image: {features_array.shape[1]}")
        
        return features_array


class LungCancerModelSelector:
    """Train multiple models and select the best one for lung cancer detection"""
    
    def __init__(self, num_classes=None, random_state=42, verbose=True):
        """
        Initialize model selector
        
        Args:
            num_classes: Number of classes (auto-detect if None)
            random_state: Random seed for reproducibility
            verbose: Print detailed output
        """
        self.random_state = random_state
        self.verbose = verbose
        self.num_classes = num_classes
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.smote = SMOTE(random_state=random_state, k_neighbors=5)
        self.models_results = {}
        
    def _init_models(self):
        """Initialize models based on number of classes"""
        
        self.models = {
            'Logistic Regression': LogisticRegression(
                max_iter=1000, 
                random_state=self.random_state, 
                class_weight='balanced',
                verbose=0
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200, 
                random_state=self.random_state, 
                class_weight='balanced_subsample',
                n_jobs=-1,
                verbose=0
            ),
            'SVM (RBF)': SVC(
                kernel='rbf', 
                probability=True, 
                class_weight='balanced',
                random_state=self.random_state,
                verbose=0
            ),
            'Gradient Boosting': GradientBoostingClassifier(
                n_estimators=200, 
                random_state=self.random_state,
                learning_rate=0.1, 
                max_depth=5,
                verbose=0
            ),
            'XGBoost': XGBClassifier(
                n_estimators=200, 
                random_state=self.random_state,
                scale_pos_weight=1 if self.num_classes > 2 else 2,
                eval_metric='mlogloss' if self.num_classes > 2 else 'logloss',
                n_jobs=-1,
                verbosity=0
            ),
            'LightGBM': LGBMClassifier(
                n_estimators=200, 
                random_state=self.random_state,
                class_weight='balanced', 
                n_jobs=-1, 
                verbose=-1
            )
        }
    
    def balance_data(self, X, y):
        """
        Balance dataset using SMOTE
        
        Args:
            X: Feature array
            y: Label array
        
        Returns:
            X_balanced: Balanced features
            y_balanced: Balanced labels
        """
        if self.verbose:
            unique, counts = np.unique(y, return_counts=True)
            print("\n CLASS DISTRIBUTION:")
            print(" BEFORE SMOTE Balancing:")
            for i, (u, c) in enumerate(zip(unique, counts)):
                percentage = (c / len(y) * 100)
                print(f"   Class {i}: {c:4} samples ({percentage:5.1f}%)")
        
        X_balanced, y_balanced = self.smote.fit_resample(X, y)
        
        if self.verbose:
            unique, counts = np.unique(y_balanced, return_counts=True)
            print("\n AFTER SMOTE Balancing:")
            for i, (u, c) in enumerate(zip(unique, counts)):
                percentage = (c / len(y_balanced) * 100)
                print(f"   Class {i}: {c:4} samples ({percentage:5.1f}%)")
        
        return X_balanced, y_balanced
    
    def evaluate_model(self, model, X_train, X_test, y_train, y_test):
        """
        Evaluate a single model
        
        Args:
            model: Model to train and evaluate
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
        
        Returns:
            Dictionary with evaluation metrics
        """
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        
        
        try:
            y_pred_proba = model.predict_proba(X_test)
            multi_class_method = 'ovr' if self.num_classes > 2 else 'ovo'
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class=multi_class_method, 
                                   average='weighted')
        except:
            roc_auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_weighted': f1,
            'roc_auc': roc_auc,
            'y_pred': y_pred,
            'y_test': y_test,
            'y_pred_proba': y_pred_proba if 'y_pred_proba' in locals() else None
        }
    
    def train_and_select(self, X, y, test_size=0.15, cv_folds=5):
        """
        Train all models with stratified cross-validation and select the best
        
        Args:
            X: Feature array
            y: Label array
            test_size: Fraction of data for testing
            cv_folds: Number of cross-validation folds
        
        Returns:
            best_model: Best performing model
            results_df: DataFrame with all results
            (X_test, y_test): Test data
        """
        
        self.num_classes = len(np.unique(y))
        print(f"\n🔍 AUTO-DETECTED NUMBER OF CLASSES: {self.num_classes}")
        
        
        self._init_models()
        
        print("\n" + "="*80)
        print("STARTING MODEL TRAINING PIPELINE")
        print("="*80)
        
        X_balanced, y_balanced = self.balance_data(X, y)
        
        print("\n  SCALING FEATURES")
        
        X_scaled = self.scaler.fit_transform(X_balanced)
        print(f"  Features scaled using StandardScaler")
        
        print("\n SPLITTING DATA")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y_balanced, test_size=test_size, stratify=y_balanced,
            random_state=self.random_state
        )
        
        if self.verbose:
            print(f"   Training data: {X_train.shape[0]} samples")
            print(f"   Test data: {X_test.shape[0]} samples")
            print(f"   Total features: {X_train.shape[1]}")
            print(f"   Test size: {test_size*100:.1f}%")
        
        
        print("\n" + "="*80)
        print(f" TRAINING AND EVALUATING {len(self.models)} ML MODELS")
        print("="*80)
        
        results_summary = []
        
        for model_idx, (model_name, model) in enumerate(self.models.items(), 1):
            print(f"\n[{model_idx}/{len(self.models)}] Training {model_name}...")
            
            try:
                
                skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, 
                                     random_state=self.random_state)
                
                cv_scores = {
                    'accuracy': [],
                    'f1_weighted': [],
                    'roc_auc': []
                }
                
                
                fold_num = 1
                for train_idx, val_idx in skf.split(X_train, y_train):
                    X_cv_train, X_cv_val = X_train[train_idx], X_train[val_idx]
                    y_cv_train, y_cv_val = y_train[train_idx], y_train[val_idx]
                    
                    model.fit(X_cv_train, y_cv_train)
                    y_pred = model.predict(X_cv_val)
                    
                    cv_scores['accuracy'].append(accuracy_score(y_cv_val, y_pred))
                    cv_scores['f1_weighted'].append(
                        f1_score(y_cv_val, y_pred, average='weighted', zero_division=0)
                    )
                    
                    try:
                        y_pred_proba = model.predict_proba(X_cv_val)
                        multi_class_method = 'ovr' if self.num_classes > 2 else 'ovo'
                        cv_scores['roc_auc'].append(
                            roc_auc_score(y_cv_val, y_pred_proba, 
                                         multi_class=multi_class_method, average='weighted')
                        )
                    except:
                        cv_scores['roc_auc'].append(0.0)
                    
                    fold_num += 1
                
                
                test_metrics = self.evaluate_model(model, X_train, X_test, 
                                                  y_train, y_test)
                
                result = {
                    'model_name': model_name,
                    'cv_accuracy_mean': np.mean(cv_scores['accuracy']),
                    'cv_accuracy_std': np.std(cv_scores['accuracy']),
                    'cv_f1_mean': np.mean(cv_scores['f1_weighted']),
                    'cv_f1_std': np.std(cv_scores['f1_weighted']),
                    'cv_roc_auc_mean': np.mean(cv_scores['roc_auc']),
                    'test_accuracy': test_metrics['accuracy'],
                    'test_precision': test_metrics['precision'],
                    'test_recall': test_metrics['recall'],
                    'test_f1': test_metrics['f1_weighted'],
                    'test_roc_auc': test_metrics['roc_auc'],
                    'model': model
                }
                
                results_summary.append(result)
                self.models_results[model_name] = result
                
                print(f"   CV F1-Score: {result['cv_f1_mean']:.4f} "
                      f"(±{result['cv_f1_std']:.4f})")
                print(f"   Test F1-Score: {result['test_f1']:.4f}")
                print(f"   Test Accuracy: {result['test_accuracy']:.4f}")
                
            except Exception as e:
                print(f"   Error training {model_name}: {e}")
                continue
        
        if not results_summary:
            raise ValueError(" No models trained successfully!")
        
        results_df = pd.DataFrame(results_summary)
        best_idx = results_df['test_f1'].idxmax()
        
        self.best_model = results_df.loc[best_idx, 'model']
        self.best_model_name = results_df.loc[best_idx, 'model_name']
        
        print("\n" + "="*80)
        print(" MODEL SELECTION RESULTS")
        print("="*80)
        print(results_df[['model_name', 'cv_f1_mean', 'test_accuracy', 
                         'test_precision', 'test_recall', 'test_f1']].to_string(index=False))
        
        print(f"\n BEST MODEL SELECTED: {self.best_model_name}")
        print(f"   Weighted F1-Score: {results_df.loc[best_idx, 'test_f1']:.4f}")
        print(f"   Accuracy: {results_df.loc[best_idx, 'test_accuracy']:.4f}")
        print(f"   Precision: {results_df.loc[best_idx, 'test_precision']:.4f}")
        print(f"   Recall: {results_df.loc[best_idx, 'test_recall']:.4f}")
        
        return self.best_model, results_df, (X_test, y_test)
    
    def save_model(self, model_path='best_model.pkl', scaler_path='scaler.pkl'):
        """Save trained model and scaler"""
        if self.best_model is None:
            raise ValueError("No model trained yet. Call train_and_select() first.")
        
        joblib.dump(self.best_model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        if self.verbose:
            print(f"\n Model Files Saved:")
            print(f"   Model: {model_path}")
            print(f"   Scaler: {scaler_path}")
    
    def export_results(self, output_path='model_results.csv'):
        """Export model results to CSV"""
        if not self.models_results:
            raise ValueError("No results to export. Call train_and_select() first.")
        
        results_data = []
        for name, result in self.models_results.items():
            results_data.append({
                'Model': name,
                'CV F1-Score': f"{result['cv_f1_mean']:.4f}",
                'CV F1-Std': f"{result['cv_f1_std']:.4f}",
                'Test F1-Score': f"{result['test_f1']:.4f}",
                'Test Accuracy': f"{result['test_accuracy']:.4f}",
                'Test Precision': f"{result['test_precision']:.4f}",
                'Test Recall': f"{result['test_recall']:.4f}",
                'Test ROC-AUC': f"{result['test_roc_auc']:.4f}"
            })
        
        results_df = pd.DataFrame(results_data)
        results_df.to_csv(output_path, index=False)
        
        if self.verbose:
            print(f" Results exported to: {output_path}")


if __name__ == "__main__":
    print("\n" + "="*80)
    print(" LUNG CANCER DETECTION SYSTEM - MODEL TRAINING PIPELINE")
    print("="*80)
    print("   data/Normal cases/")
    print("   data/Benign cases/ ")
    print("   data/Malignant cases/")
    
   
    print("\n" + "="*80)
    print("STEP 1: Loading CT Scan Images")
    print("="*80)
    
    loader = LungCancerDataLoader(data_dir='data/', image_size=(128, 128))
    X, y = loader.load_images(verbose=True)
    
    if len(X) == 0:
        print("\nERROR: No images loaded!")
        print("Please verify your data directory structure:")
        print("   data/")
        print("   ├── Normal cases/")
        print("   ├── Benign cases/")
        print("   └── Malignant cases/")
        exit(1)
    
    
    print("\n" + "="*80)
    print("STEP 2: Extracting Features")
    print("="*80)
    
    X = loader.extract_features(X, verbose=True)
    
    print("\n" + "="*80)
    print("STEP 3: Training and Selecting Best Model")
    print("="*80)
    
    selector = LungCancerModelSelector(random_state=42, verbose=True)
    best_model, results_df, (X_test, y_test) = selector.train_and_select(
        X, y, test_size=0.15, cv_folds=5
    )
    
    print("\n" + "="*80)
    print("STEP 4: Saving Model and Scaler")
    print("="*80)
    
    selector.save_model(model_path='best_model.pkl', scaler_path='scaler.pkl')
    
    print("\n" + "="*80)
    print("STEP 5: Exporting Results")
    print("="*80)
    
    selector.export_results(output_path='model_results.csv')
    
    print("\n" + "="*80)
    print("TRAINING COMPLETE!")
    print("="*80)
    print(f"\n Best Model: {selector.best_model_name}")
    print(f"\n Files created:")
    print(f"   1. best_model.pkl (trained model)")
    print(f"   2. scaler.pkl (feature scaler)")
    print(f"   3. model_results.csv (results summary)")
    print(f"\n To run Streamlit app: streamlit run streamlit_app.py")
    print("\n" + "="*80)