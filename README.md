# Student Performance Prediction - End-to-End ML Project

A complete machine learning pipeline that predicts student math scores based on demographic and academic features, deployed as a web application on AWS Elastic Beanstalk.

## 🎯 Project Overview

This project demonstrates a full ML lifecycle from data ingestion to production deployment:
- Exploratory Data Analysis (EDA) on student performance data
- Feature engineering and preprocessing pipeline
- Model training with multiple algorithms
- Model evaluation and selection
- Flask web application for predictions
- CI/CD deployment on AWS Elastic Beanstalk

## 📊 Dataset & Features

**Target Variable:** `math_score` (0-100)

**Input Features:**
- `gender`: Student gender (male/female)
- `race_ethnicity`: Ethnic group (group A-E)
- `parental_level_of_education`: Parents' highest education level
- `lunch`: Lunch type (standard/free or reduced)
- `test_preparation_course`: Completion status (none/completed)
- `reading_score`: Reading test score (0-100)
- `writing_score`: Writing test score (0-100)

**Dataset Split:**
- Training: 800 samples (80%)
- Testing: 200 samples (20%)

## 🔍 Data Challenges

### 1. **Mixed Data Types**
- Categorical features (5): gender, race/ethnicity, education, lunch, test prep
- Numerical features (2): reading and writing scores
- **Solution:** Built separate preprocessing pipelines using `ColumnTransformer`

### 2. **Categorical Encoding**
- High-cardinality categorical variables with no inherent order
- **Solution:** One-hot encoding with `drop='first'` to avoid multicollinearity and `handle_unknown='ignore'` for robustness

### 3. **Feature Scaling**
- Different scales between numeric features and one-hot encoded features
- **Solution:** StandardScaler for numeric features; no scaling for categorical dummies to preserve interpretability

### 4. **Missing Values**
- Potential missing data in both numeric and categorical columns
- **Solution:** SimpleImputer with median strategy for numeric, most_frequent for categorical

## 🏗️ Project Architecture

```
ML_project/
├── src/
│   ├── components/          # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py
│   │   └── model_evaluation.py
│   ├── pipeline/            # Pipeline orchestration
│   │   ├── stage_01_data_ingestion.py
│   │   ├── stage_02_data_transformation.py
│   │   ├── stage_03_model_trainer.py
│   │   └── stage_04_model_evaluation.py
│   ├── utils/               # Utility functions
│   ├── logged/              # Logging configuration
│   └── exception.py         # Custom exception handling
├── artifacts/               # Generated outputs
│   ├── train.csv
│   ├── test.csv
│   ├── preprocessor.pkl
│   ├── model.pkl
│   └── evaluation_report.json
├── templates/               # HTML templates
├── logs/                    # Application logs
├── application.py           # Flask app (AWS EB entry point)
├── main.py                  # Pipeline orchestrator
└── requirements.txt
```

## 🔄 ML Pipeline

### Stage 1: Data Ingestion
- Reads raw CSV data
- Splits into train/test sets (80/20)
- Saves artifacts to `artifacts/` directory

### Stage 2: Data Transformation
- **Numeric Pipeline:**
  - Imputation: Median strategy
  - Scaling: StandardScaler (zero mean, unit variance)
- **Categorical Pipeline:**
  - Imputation: Most frequent strategy
  - Encoding: OneHotEncoder with reference category
- Saves fitted preprocessor to `artifacts/preprocessor.pkl`

### Stage 3: Model Training
Trained and evaluated multiple regression models:
- Linear Regression
- K-Nearest Neighbors
- Decision Tree
- Random Forest
- Gradient Boosting
- AdaBoost
- XGBoost (optional)
- CatBoost (optional)

**Best Model Selection:** Automatically selects model with highest R² score

### Stage 4: Model Evaluation
- Computes metrics on test set
- Saves evaluation report with R², MAE, MSE, RMSE
- Logs detailed performance metrics

## 📈 Model Performance

**Best Model:** Linear Regression

| Metric | Value |
|--------|-------|
| R² Score | 0.8804 |
| MAE | 4.21 |
| RMSE | 5.39 |
| Test Samples | 200 |

**Model Interpretation:**
- Strong positive correlation with writing score (coefficient: ~10.92)
- Moderate positive correlation with reading score (coefficient: ~3.41)
- Gender, lunch type, and test preparation show significant effects
- Model explains ~88% of variance in math scores

## 🚀 Deployment

### Local Development
```bash
# Setup virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the full pipeline
python main.py

# Start Flask app
PORT=8080 python application.py
```

Access at: `http://localhost:8080`

### AWS Elastic Beanstalk Deployment

**Platform:** Python 3.13 on Amazon Linux 2023

#### **Quick Deploy:**
```bash
pip install awsebcli && aws configure
eb init -p python-3.13 student-predict --region us-east-1
eb create student-predict-env --instance-type t2.micro
eb open
```

#### **Key Challenge & Solution:**
- **Problem:** Deployment failed due to heavy packages (catboost, xgboost) requiring compilation
- **Solution:** Removed training packages from `requirements.txt` - model is pre-trained, only needs prediction dependencies

**Production packages:** flask, pandas, numpy, scikit-learn, gunicorn, dill

---

### **CI/CD with GitHub Actions**

**Auto-deploy on push to `main`**

1. Add AWS credentials to GitHub Secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
2. Push code → GitHub Actions deploys automatically (~5 min)

**Files:** `Procfile`, `.ebignore`, `.github/workflows/deploy.yml`

## 🛠️ Technologies Used

**ML & Data Science:**
- scikit-learn: Preprocessing, modeling, evaluation
- pandas: Data manipulation
- numpy: Numerical operations
- dill/pickle: Model serialization

**Web Framework:**
- Flask: Web application and API
- Jinja2: HTML templating

**Deployment:**
- AWS Elastic Beanstalk: Application hosting
- AWS EC2: Compute instances
- AWS S3: Artifact storage (optional)

**Development & DevOps:**
- Python 3.13
- Git & GitHub: Version control
- GitHub Actions: CI/CD automation
- Logging: Custom logging framework

## 📝 Usage

### Web Interface
1. Navigate to the deployed URL or `http://localhost:8080`
2. Fill in the student information form:
   - Select demographic details
   - Enter reading and writing scores
3. Click "Predict" to get the predicted math score

### Programmatic Usage
```python
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import pandas as pd

# Load and transform data
transformer = DataTransformation()
train_arr, test_arr, _ = transformer.initiate_data_transformation(
    'artifacts/train.csv', 
    'artifacts/test.csv'
)

# Train model
trainer = ModelTrainer()
name, score, path = trainer.initiate_model_training(train_arr, test_arr)
print(f"Best Model: {name} with R²={score:.4f}")
```

## 🔧 Configuration

**Logging:**
- Logs stored in `logs/` directory
- Timestamped log files for each run
- Configurable via `src/logged/logger.py`

**Model Artifacts:**
- Preprocessor: `artifacts/preprocessor.pkl`
- Model: `artifacts/model.pkl`
- Evaluation: `artifacts/evaluation_report.json`

## 🧪 Testing

Run individual pipeline stages:
```bash
# Data ingestion
python src/pipeline/stage_01_data_ingestion.py

# Data transformation
python src/pipeline/stage_02_data_transformation.py

# Model training
python src/pipeline/stage_03_model_trainer.py

# Model evaluation
python src/pipeline/stage_04_model_evaluation.py
```

## 📊 Key Insights

1. **Writing score is the strongest predictor** of math performance (3x stronger than reading)
2. **Test preparation course completion** shows positive impact on scores
3. **Lunch type** (proxy for socioeconomic status) significantly affects performance
4. **Linear model performs best**, suggesting linear relationships in the data
5. **Model generalizes well** with consistent train/test performance

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Hyperparameter tuning with GridSearchCV
- Feature engineering (interaction terms, polynomial features)
- Model ensembling and stacking
- API rate limiting and authentication
- Unit tests and integration tests
- Load testing and performance optimization

## 📄 License

This project is open source and available under the MIT License.

## 👤 Author

**Omkar Thakur**
- Email: othakur@umd.edu
- GitHub: [Info-stats-ai/Machine_learning_Project](https://github.com/Info-stats-ai/Machine_learning_Project)

## 🙏 Acknowledgments

- Dataset: Student Performance Dataset
- Deployment: AWS Elastic Beanstalk
- Framework: scikit-learn, Flask

---

**Project Status:** ✅ Production Ready with CI/CD

**Last Updated:** October 29, 2025
