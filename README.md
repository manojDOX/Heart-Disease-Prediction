# Heart Disease Prediction Project 🫀

A comprehensive machine learning project for predicting heart disease using multiple algorithms with automated model comparison and selection.

## 📋 Project Overview

This project implements a complete machine learning pipeline to predict heart disease based on patient medical data. The system automatically trains multiple models, evaluates their performance, and selects the best performing model for deployment.

### 🎯 Key Features

- **Multi-Model Training**: Implements XGBoost, Random Forest, and Decision Tree algorithms
- **Automated Feature Selection**: Uses SelectFromModel for optimal feature engineering
- **Hyperparameter Optimization**: GridSearchCV for finding best model parameters
- **Comprehensive Evaluation**: Multiple metrics including accuracy, precision, recall, F1-score, and AUC
- **Model Persistence**: Automatic saving of trained models and scalers
- **Visual Analytics**: ROC curves and confusion matrices for model interpretation

## 🏗️ Project Structure

```
heart_disease_prediction/
│
├── data/
│   └── heart.csv                          # Heart disease dataset
│
├── scripts/
│   ├── __pycache__/                       # Python cache files
│   ├── data_loader.py                     # Data loading and preprocessing
│   ├── evaluate_models.py                 # Model evaluation utilities
│   ├── main.py                           # Main execution pipeline
│   ├── preprocessing.py                   # Data preprocessing functions
│   ├── requirements.txt                   # Project dependencies
│   └── train_models.py                    # Model training implementations
│
├── models/                                # Trained model files (generated)
│   ├── best_model.pkl                     # Best performing model
│   ├── decisiontree_model.pkl            # Decision Tree model
│   ├── randomforest_model.pkl            # Random Forest model
│   ├── scaler.pkl                        # Feature scaler
│   ├── selected_features.txt             # Selected feature names
│   └── xgboost_model.pkl                 # XGBoost model
│
├── model_graphs/                          # Generated visualizations
│   ├── Decisiontree_confusion_matrix.png # Decision Tree confusion matrix
│   ├── Decisiontree_roc_curve.png       # Decision Tree ROC curve
│   ├── Randomforest_confusion_matrix.png # Random Forest confusion matrix
│   ├── Randomforest_roc_curve.png       # Random Forest ROC curve
│   ├── Xgboost_confusion_matrix.png     # XGBoost confusion matrix
│   └── Xgboost_roc_curve.png            # XGBoost ROC curve
│
└── env/                                   # Virtual environment
```

## 📊 Dataset Information

The project uses the heart disease dataset with the following features:

| Feature | Description | Type |
|---------|-------------|------|
| Age | Age of the patient | Numerical |
| Sex | Gender (M/F) | Categorical |
| ChestPainType | Type of chest pain (ATA/NAP/ASY/TA) | Categorical |
| RestingBP | Resting blood pressure | Numerical |
| Cholesterol | Serum cholesterol level | Numerical |
| FastingBS | Fasting blood sugar | Binary |
| RestingECG | Resting electrocardiogram | Categorical |
| MaxHR | Maximum heart rate achieved | Numerical |
| ExerciseAngina | Exercise induced angina | Binary |
| Oldpeak | ST depression induced by exercise | Numerical |
| ST_Slope | Slope of peak exercise ST segment | Categorical |
| HeartDisease | Target variable (0/1) | Binary |

**Dataset Statistics:**
- Total samples: 918
- Features: 12
- No missing values
- No duplicate records

## 🤖 Machine Learning Pipeline

### 1. Data Preprocessing
- **Categorical Encoding**: One-hot encoding for categorical variables
- **Feature Selection**: SelectFromModel reduces features from 15 to 8 most important ones
- **Data Scaling**: StandardScaler for numerical feature normalization
- **Train-Test Split**: 80-20 split (734 training, 184 testing samples)

### 2. Model Training
The pipeline trains three different algorithms with hyperparameter tuning:

#### XGBoost Classifier
- **Best Parameters**: 
  - Learning Rate: 0.05
  - Max Depth: 5
  - N Estimators: 100
  - Subsample: 0.8

#### Random Forest Classifier
- **Best Parameters**:
  - Max Depth: 5
  - Min Samples Split: 5
  - N Estimators: 100

#### Decision Tree Classifier
- **Best Parameters**:
  - Max Depth: 3
  - Min Samples Split: 2

### 3. Model Evaluation

| Model | Accuracy | Precision | Recall | F1-Score | AUC |
|-------|----------|-----------|---------|----------|-----|
| **Random Forest** | **84.24%** | **82.88%** | **90.20%** | **86.38%** | **90.77%** |
| XGBoost | 82.07% | 82.24% | 86.27% | 84.21% | 89.36% |
| Decision Tree | 79.89% | 78.76% | 87.25% | 82.79% | 83.73% |

🏆 **Best Model**: Random Forest (automatically selected and saved)

## 📈 Selected Features

The feature selection process identified these 8 most important features:
- Age
- RestingBP
- Cholesterol
- MaxHR
- Oldpeak
- ExerciseAngina_Y
- ST_Slope_Flat
- ST_Slope_Up

## 🔧 Technical Implementation

### Key Technologies
- **Python 3.x**: Core programming language
- **Scikit-learn**: Machine learning framework
- **XGBoost**: Gradient boosting implementation
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **Matplotlib/Seaborn**: Data visualization
- **Joblib**: Model serialization

### Design Patterns
- **Modular Architecture**: Separate modules for data loading, preprocessing, training, and evaluation
- **Configuration-Driven**: Hyperparameter grids for systematic tuning
- **Automated Pipeline**: End-to-end execution with minimal manual intervention
- **Model Persistence**: Automatic saving of trained models and preprocessing objects

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.7 or higher
- Git (for cloning the repository)

### Installation & Setup

1. **Clone the Repository**
   ```bash
   git clone <your-repository-url>
   cd heart_disease_prediction
   ```

2. **Remove Generated Folders** (if present)
   ```bash
   # Remove model and graph folders to start fresh
   rm -rf models/
   rm -rf model_graphs/
   ```

3. **Create Virtual Environment**
   ```bash
   # Windows
   python -m venv env
   
   # macOS/Linux
   python3 -m venv env
   ```

4. **Activate Virtual Environment**
   ```bash
   # Windows
   env\Scripts\activate
   
   # macOS/Linux
   source env/bin/activate
   ```

5. **Install Dependencies**
   ```bash
   pip install -r scripts/requirements.txt
   ```

6. **Run the Complete Pipeline**
   ```bash
   python scripts/main.py
   ```

### Expected Output

When you run the pipeline, you should see:
- ✅ Data loading confirmation
- 📊 Dataset overview and statistics
- 🎯 Feature selection results
- 🏋️ Model training progress
- 📈 Comprehensive model evaluation
- 🏆 Best model identification and saving

The system will automatically generate:
- **Trained Models**: Saved in `models/` directory
- **Visualizations**: ROC curves and confusion matrices in `model_graphs/`
- **Feature Information**: Selected features list in `selected_features.txt`

## 📁 Generated Files

After running the pipeline, the following files will be created:

### Models Directory
- `best_model.pkl` - The highest performing model
- `xgboost_model.pkl` - XGBoost classifier
- `randomforest_model.pkl` - Random Forest classifier  
- `decisiontree_model.pkl` - Decision Tree classifier
- `scaler.pkl` - Fitted StandardScaler
- `selected_features.txt` - List of selected features

### Model Graphs Directory
- ROC curves for each model
- Confusion matrices for each model
- Performance comparison visualizations

## 🔍 Model Performance Analysis

The Random Forest model achieved the best overall performance with:
- **High Accuracy**: 84.24% correct predictions
- **Balanced Performance**: Good precision-recall balance
- **Excellent AUC**: 90.77% area under ROC curve
- **Clinical Relevance**: High recall (90.20%) minimizes false negatives

## 🛠️ Customization Options

### Modifying Hyperparameters
Edit the parameter grids in `train_models.py`:
```python
xgb_params = {
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'n_estimators': [50, 100, 200]
}
```

### Adding New Models
Extend the pipeline by adding new algorithms in `train_models.py` and updating the evaluation loop in `main.py`.

### Feature Engineering
Modify `preprocessing.py` to add new feature transformations or selection strategies.

## 🎯 Use Cases

- **Medical Research**: Analyzing risk factors for heart disease
- **Clinical Decision Support**: Assisting healthcare professionals
- **Preventive Healthcare**: Identifying high-risk patients
- **Educational**: Learning machine learning concepts and implementation

## 📋 Requirements

See `scripts/requirements.txt` for complete dependency list. Key packages include:
- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- joblib

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the complete pipeline
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Heart disease dataset from [UCI Machine Learning Repository]
- Scikit-learn community for excellent ML tools
- XGBoost developers for gradient boosting implementation

## 📞 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. For medical applications, always consult qualified healthcare professionals.
