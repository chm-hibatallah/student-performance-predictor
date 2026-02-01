# 🎓 Student Exam Performance Predictor

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-2.3.0-green.svg)](https://flask.palletsprojects.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3.0-orange.svg)](https://scikit-learn.org/)
[![Deployed on Render](https://img.shields.io/badge/Deployed%20on-Render-46E3B7.svg)](https://ml-project-ed15.onrender.com)

A complete end-to-end Machine Learning project that predicts student math scores based on various demographic and educational factors. The project demonstrates the full ML lifecycle from data exploration to model deployment.

## 🌐 Live Demo

**Check out the live application:** [https://ml-project-ed15.onrender.com](https://ml-project-ed15.onrender.com)

---

## 📋 Table of Contents

- [Problem Statement](#-problem-statement)
- [Project Architecture](#-project-architecture)
- [Dataset Information](#-dataset-information)
- [Exploratory Data Analysis](#-exploratory-data-analysis)
- [Data Pipeline](#-data-pipeline)
- [Model Training & Evaluation](#-model-training--evaluation)
- [Hyperparameter Tuning](#-hyperparameter-tuning)
- [Prediction Pipeline](#-prediction-pipeline)
- [Web Application](#-web-application)
- [Deployment](#-deployment)
- [Installation & Setup](#-installation--setup)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Key Features](#-key-features)
- [Future Improvements](#-future-improvements)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Problem Statement

This project aims to understand how student performance (test scores) is affected by various factors including:
- **Gender**
- **Ethnicity/Race**
- **Parental Level of Education**
- **Lunch Type** (standard vs free/reduced)
- **Test Preparation Course** (completed vs none)
- **Reading Score**
- **Writing Score**

The goal is to predict a student's **Math Score** based on these input features using machine learning regression models.

### Why This Matters
Understanding these factors can help:
- Educators identify students who may need additional support
- Policy makers allocate resources effectively
- Parents understand the impact of various educational factors
- Students optimize their learning strategies

---

## 🏗️ Project Architecture

This is a complete **end-to-end ML project** with the following components:

```
Data Collection → EDA → Data Ingestion → Data Transformation → 
Model Training → Model Evaluation → Hyperparameter Tuning → 
Prediction Pipeline → Flask Web App → Deployment
```

### Key Components:

1. **Data Ingestion**: Automated data loading and splitting
2. **Data Transformation**: Feature engineering and preprocessing
3. **Model Training**: Training multiple regression models
4. **Model Evaluation**: Comprehensive performance metrics
5. **Prediction Pipeline**: Real-time prediction system
6. **Web Interface**: User-friendly Flask application
7. **Deployment**: Production-ready deployment on Render

---

## 📊 Dataset Information

- **Source**: [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Size**: 1,000 rows × 8 columns
- **Target Variable**: Math Score
- **Features**: 7 (5 categorical, 2 numerical)

### Features Description:

| Feature | Type | Description |
|---------|------|-------------|
| `gender` | Categorical | Student's gender (male/female) |
| `race_ethnicity` | Categorical | Student's ethnicity (Group A-E) |
| `parental_level_of_education` | Categorical | Highest education level of parents |
| `lunch` | Categorical | Type of lunch (standard/free-reduced) |
| `test_preparation_course` | Categorical | Completion status of test prep course |
| `reading_score` | Numerical | Score in reading (0-100) |
| `writing_score` | Numerical | Score in writing (0-100) |
| `math_score` | Numerical | **Target**: Score in math (0-100) |

---

## 🔍 Exploratory Data Analysis

### Key Insights from EDA:

#### 1. **Gender Impact**
- **Females** have higher pass percentages overall
- Females are top scorers in reading and writing
- Gender shows correlation with performance across all subjects

#### 2. **Parental Education**
- Strong positive correlation between parental education level and student scores
- Students with parents holding bachelor's or master's degrees score significantly higher

#### 3. **Lunch Type**
- Students with **standard lunch** perform better than those with free/reduced lunch
- This indicates potential socioeconomic factors affecting performance

#### 4. **Test Preparation Course**
- Completing the test preparation course shows moderate benefit
- Students who completed prep courses show improved scores

#### 5. **Score Correlations**
- **Strong linear correlation** between reading, writing, and math scores
- All three scores tend to increase together

#### 6. **Ethnicity Factor**
- Performance varies across different ethnic groups
- Group E shows highest average performance

### Visualizations Created:
- Distribution plots for all scores
- Correlation heatmaps
- Box plots for categorical features
- Pair plots showing relationships between variables
- Count plots for categorical distributions

---

## 🔄 Data Pipeline

### 1. Data Ingestion Component

**File**: `src/components/data_ingestion.py`

**Responsibilities**:
- Loads raw data from source
- Splits data into train and test sets (80:20 ratio)
- Saves train/test datasets to artifacts folder
- Returns paths for further processing

```python
# Key functionality
- Read CSV data
- Train-test split (test_size=0.2, random_state=42)
- Save processed datasets
```

### 2. Data Transformation Component

**File**: `src/components/data_transformation.py`

**Responsibilities**:
- **Numerical Features**: StandardScaler normalization
- **Categorical Features**: OneHotEncoding + StandardScaler
- Creates preprocessing pipeline
- Saves preprocessor as pickle file

**Preprocessing Steps**:

| Feature Type | Transformations |
|--------------|----------------|
| Numerical | `StandardScaler` |
| Categorical | `OneHotEncoder` → `StandardScaler` |

**Categorical Features Processed**:
- gender
- race_ethnicity
- parental_level_of_education
- lunch
- test_preparation_course

**Numerical Features Processed**:
- reading_score
- writing_score

---

## 🤖 Model Training & Evaluation

### Models Trained

Nine different regression algorithms were trained and evaluated:

| # | Model | R² Score (Test) | RMSE | MAE |
|---|-------|----------------|------|-----|
| 1 | **Linear Regression** ⭐ | **0.8804** | **5.4052** | **4.2680** |
| 2 | **Ridge Regression** ⭐ | **0.8806** | **5.4018** | **4.2649** |
| 3 | Random Forest Regressor | 0.8538 | 5.9644 | 4.5901 |
| 4 | CatBoost Regressor | 0.8516 | 6.0086 | 4.6125 |
| 5 | AdaBoost Regressor | 0.8496 | 6.0494 | 4.6677 |
| 6 | XGBoost Regressor | 0.8278 | 6.4733 | 5.0577 |
| 7 | Lasso Regression | 0.8253 | 6.5259 | 5.1515 |
| 8 | K-Neighbors Regressor | 0.7837 | 7.2649 | 5.8225 |
| 9 | Decision Tree | 0.7356 | 8.0299 | 6.3972 |

### Best Performing Models

**🏆 Top 2 Models: Ridge Regression & Linear Regression**

Both models achieved approximately **88% R² Score**, indicating they can explain 88% of the variance in math scores.

#### Why Ridge Regression was Selected:
- Highest R² Score (0.8806)
- Low RMSE (5.40) - predictions are within ~5.4 points on average
- Good generalization (minimal overfitting)
- Regularization prevents overfitting better than Linear Regression

### Evaluation Metrics Explained:

- **R² Score**: Proportion of variance explained (higher is better, max 1.0)
- **RMSE** (Root Mean Squared Error): Average prediction error in score points (lower is better)
- **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual scores (lower is better)

---

## ⚙️ Hyperparameter Tuning

### Approach Used:
- **GridSearchCV** for exhaustive parameter search
- **5-Fold Cross-Validation** for robust evaluation
- **R² Score** as optimization metric

### Models Tuned:

#### 1. **Ridge Regression** (Selected Model)
```python
Parameters tuned:
- alpha: [0.01, 0.1, 1, 10, 100]
- solver: ['auto', 'svd', 'cholesky', 'lsqr']

Best Parameters:
- alpha: 100
- solver: 'auto'
```

#### 2. **Random Forest**
```python
Parameters tuned:
- n_estimators: [8, 16, 32, 64, 128, 256]
- max_depth: [None, 10, 20, 30]
- min_samples_split: [2, 5, 10]
```

#### 3. **AdaBoost**
```python
Parameters tuned:
- n_estimators: [8, 16, 32, 64, 128, 256]
- learning_rate: [0.01, 0.1, 0.5, 1.0]
```

### Cross-Validation Strategy:
- 5-fold cross-validation
- Ensures model generalizes well to unseen data
- Reduces overfitting risk

---

## 🔮 Prediction Pipeline

**File**: `src/pipeline/predict_pipeline.py`

### Components:

#### 1. **CustomData Class**
Handles input data transformation:
```python
class CustomData:
    - Accepts user inputs
    - Converts to pandas DataFrame
    - Ensures correct feature format
```

#### 2. **PredictPipeline Class**
Manages prediction process:
```python
class PredictPipeline:
    - Loads trained model
    - Loads preprocessor
    - Transforms input data
    - Returns predictions
```

### Prediction Flow:
```
User Input → CustomData → DataFrame → 
Preprocessor → Transformed Features → 
Model → Prediction → User
```

---

## 🌐 Web Application

### Technology Stack:
- **Framework**: Flask 2.3.0
- **Frontend**: HTML5, CSS3
- **Backend**: Python 3.8+

### Features:

#### 1. **Landing Page** (`/`)
- Professional welcome interface
- Project description
- "Start Prediction" button
- Feature highlights

#### 2. **Prediction Page** (`/predictdata`)
- User-friendly form with:
  - Gender selection
  - Ethnicity dropdown
  - Parental education level
  - Lunch type
  - Test prep course status
  - Reading score input (0-100)
  - Writing score input (0-100)
- Real-time validation
- Clear error messages
- Instant predictions

#### 3. **Results Display**
- Predicted math score (0-100 scale)
- Clean, professional presentation
- Option to make new predictions

### Application Routes:

```python
@app.route('/')
def index():
    # Landing page
    
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    # Prediction form and results
```

### UI/UX Features:
- Responsive design
- Gradient backgrounds
- Smooth transitions
- Input validation
- Clear labels and placeholders
- Professional color scheme

---

## 🚀 Deployment

### Platform: **Render**

**Live URL**: [https://ml-project-ed15.onrender.com](https://ml-project-ed15.onrender.com)

### Deployment Process:

#### 1. **Preparation**
- Updated `requirements.txt` with all dependencies
- Configured Flask app for production
- Set environment variable for PORT
- Removed AWS-specific files

#### 2. **Configuration**
```python
# app.py configuration
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
```

#### 3. **Render Settings**
- **Build Command**: `pip install -r requirements.txt`
- **Start Command**: `python app.py`
- **Python Version**: 3.10
- **Instance Type**: Free tier

#### 4. **Continuous Deployment**
- Connected to GitHub repository
- Auto-deploys on git push to main branch
- Build logs available for debugging

### Production Considerations:
- Debug mode disabled
- Error handling implemented
- Logging system in place
- Exception handling throughout

---

## 💻 Installation & Setup

### Prerequisites:
- Python 3.8 or higher
- pip package manager
- Git

### Local Installation:

#### 1. **Clone the Repository**
```bash
git clone https://github.com/chm-hibatallah/ML-Project.git
cd ML-Project
```

#### 2. **Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

#### 3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

#### 4. **Run the Application**
```bash
python app.py
```

#### 5. **Access the Application**
Open your browser and navigate to:
```
http://localhost:5000
```

### Requirements File:
```txt
Flask==2.3.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
catboost==1.2
seaborn==0.12.2
matplotlib==3.7.2
```

---

## 📖 Usage

### Making Predictions:

#### Via Web Interface:

1. **Navigate to the application**
   - Visit: https://ml-project-ed15.onrender.com
   - Or run locally: http://localhost:5000

2. **Click "Start Prediction"**

3. **Fill in the form**:
   - Select gender
   - Choose ethnicity group
   - Select parental education level
   - Choose lunch type
   - Select test prep course status
   - Enter reading score (0-100)
   - Enter writing score (0-100)

4. **Click "Predict Math Score"**

5. **View Results**:
   - Predicted math score displayed
   - Make another prediction or go back

#### Example Input:
```
Gender: Female
Ethnicity: Group B
Parental Education: Bachelor's Degree
Lunch: Standard
Test Prep: Completed
Reading Score: 72
Writing Score: 74
```

#### Example Output:
```
Predicted Math Score: 73.45/100
```

---

## 📁 Project Structure

```
ML-Project/
│
├── src/
│   ├── components/
│   │   ├── __init__.py
│   │   ├── data_ingestion.py       # Data loading and splitting
│   │   ├── data_transformation.py  # Feature preprocessing
│   │   └── model_trainer.py        # Model training logic
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── predict_pipeline.py     # Prediction pipeline
│   │   └── train_pipeline.py       # Training pipeline
│   │
│   ├── exception.py                # Custom exception handling
│   ├── logger.py                   # Logging configuration
│   └── utils.py                    # Utility functions
│
├── artifacts/
│   ├── model.pkl                   # Trained model
│   ├── preprocessor.pkl            # Data preprocessor
│   ├── train.csv                   # Training data
│   └── test.csv                    # Testing data
│
├── notebook/
│   ├── 1_EDA_STUDENT_PERFORMANCE.ipynb    # Exploratory analysis
│   └── 2_MODEL_TRAINING.ipynb             # Model experiments
│
├── templates/
│   ├── index.html                  # Landing page
│   └── home.html                   # Prediction form
│
├── app.py                          # Flask application
├── setup.py                        # Package setup
├── requirements.txt                # Dependencies
├── README.md                       # This file
└── .gitignore                      # Git ignore rules
```

### Key Directories:

- **`src/`**: Source code for the ML pipeline
- **`artifacts/`**: Saved models and preprocessed data
- **`notebook/`**: Jupyter notebooks for EDA and experimentation
- **`templates/`**: HTML templates for web interface

---

## 🛠️ Technologies Used

### Core Technologies:

#### Machine Learning & Data Science:
- ![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) **Python 3.8+**
- ![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white) **scikit-learn** - ML algorithms
- ![Pandas](https://img.shields.io/badge/Pandas-150458?logo=pandas&logoColor=white) **Pandas** - Data manipulation
- ![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white) **NumPy** - Numerical computing

#### Web Development:
- ![Flask](https://img.shields.io/badge/Flask-000000?logo=flask&logoColor=white) **Flask** - Web framework
- ![HTML5](https://img.shields.io/badge/HTML5-E34F26?logo=html5&logoColor=white) **HTML5** - Structure
- ![CSS3](https://img.shields.io/badge/CSS3-1572B6?logo=css3&logoColor=white) **CSS3** - Styling

#### Visualization:
- **Matplotlib** - Static plots
- **Seaborn** - Statistical visualizations

#### ML Models Used:
- Linear Regression
- Ridge Regression ⭐ (Selected)
- Lasso Regression
- Random Forest
- XGBoost
- CatBoost
- AdaBoost
- K-Nearest Neighbors
- Decision Tree

#### Development Tools:
- ![Git](https://img.shields.io/badge/Git-F05032?logo=git&logoColor=white) **Git** - Version control
- ![GitHub](https://img.shields.io/badge/GitHub-181717?logo=github&logoColor=white) **GitHub** - Code hosting
- ![Jupyter](https://img.shields.io/badge/Jupyter-F37626?logo=jupyter&logoColor=white) **Jupyter** - Notebooks
- ![VS Code](https://img.shields.io/badge/VS_Code-007ACC?logo=visual-studio-code&logoColor=white) **VS Code** - IDE

#### Deployment:
- ![Render](https://img.shields.io/badge/Render-46E3B7?logo=render&logoColor=white) **Render** - Cloud platform

---

## ✨ Key Features

### 1. **Complete ML Pipeline**
- ✅ Automated data ingestion
- ✅ Robust data transformation
- ✅ Multiple model training
- ✅ Comprehensive evaluation
- ✅ Production-ready predictions

### 2. **Professional Code Structure**
- ✅ Modular architecture
- ✅ Custom exception handling
- ✅ Comprehensive logging
- ✅ Reusable components
- ✅ Clean code principles

### 3. **Robust Error Handling**
- ✅ Custom exception classes
- ✅ Detailed error messages
- ✅ Logging throughout pipeline
- ✅ User-friendly error displays

### 4. **Production Ready**
- ✅ Deployed on cloud (Render)
- ✅ Environment configuration
- ✅ Scalable architecture
- ✅ Continuous deployment

### 5. **User-Friendly Interface**
- ✅ Intuitive web design
- ✅ Input validation
- ✅ Clear instructions
- ✅ Responsive layout
- ✅ Professional styling

### 6. **Documentation**
- ✅ Comprehensive README
- ✅ Code comments
- ✅ Docstrings
- ✅ Usage examples

---

## 🔄 Logging and Exception Handling

### Custom Logger

**File**: `src/logger.py`

**Features**:
- Timestamped log files
- Formatted log messages
- Multiple log levels (INFO, ERROR, DEBUG)
- Saved in `logs/` directory

**Log Format**:
```
[2024-02-01 15:30:45,123] - module_name - INFO - Log message
```

### Custom Exception Handler

**File**: `src/exception.py`

**Features**:
- Detailed error messages
- File name and line number tracking
- Stack trace preservation
- User-friendly error formatting

**Exception Format**:
```
Error occurred in python script: [filename]
Line number: [line_no]
Error message: [error_message]
```

### Implementation Example:
```python
try:
    # Code that might raise an exception
    result = some_function()
except Exception as e:
    logging.info("Error occurred in data ingestion")
    raise CustomException(e, sys)
```

---

## 🚧 Future Improvements

### Planned Enhancements:

#### 1. **Model Improvements**
- [ ] Implement ensemble methods
- [ ] Add deep learning models
- [ ] Feature engineering experiments
- [ ] Try different feature selection techniques

#### 2. **Application Features**
- [ ] Add user authentication
- [ ] Save prediction history
- [ ] Batch predictions via CSV upload
- [ ] Download prediction reports
- [ ] Add data visualization on results page

#### 3. **Technical Enhancements**
- [ ] Add API endpoints (REST API)
- [ ] Implement caching for predictions
- [ ] Add model versioning
- [ ] Create A/B testing framework
- [ ] Add monitoring and analytics

#### 4. **Deployment**
- [ ] Add CI/CD pipeline
- [ ] Implement automated testing
- [ ] Add Docker containerization
- [ ] Set up database for predictions
- [ ] Add load balancing

#### 5. **Documentation**
- [ ] Add API documentation
- [ ] Create video tutorials
- [ ] Add more usage examples
- [ ] Create developer guide

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

### How to Contribute:

1. **Fork the repository**
2. **Create a feature branch**
   ```bash
   git checkout -b feature/AmazingFeature
   ```
3. **Commit your changes**
   ```bash
   git commit -m 'Add some AmazingFeature'
   ```
4. **Push to the branch**
   ```bash
   git push origin feature/AmazingFeature
   ```
5. **Open a Pull Request**

### Contribution Guidelines:
- Follow PEP 8 style guide
- Add tests for new features
- Update documentation
- Ensure all tests pass
- Write clear commit messages

---

## 📞 Contact

**Project Author**: Chm Hibatallah

- **GitHub**: [@chm-hibatallah](https://github.com/chm-hibatallah)
- **Project Link**: [https://github.com/chm-hibatallah/ML-Project](https://github.com/chm-hibatallah/ML-Project)
- **Live Demo**: [https://ml-project-ed15.onrender.com](https://ml-project-ed15.onrender.com)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **Dataset**: [Kaggle - Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams)
- **Flask Documentation**: [Flask Official Docs](https://flask.palletsprojects.com/)
- **scikit-learn**: [scikit-learn Documentation](https://scikit-learn.org/)
- **Render**: [Render Documentation](https://render.com/docs)

---

## 📊 Project Statistics

- **Lines of Code**: ~2,000+
- **Number of Models Trained**: 9
- **Best Model Accuracy**: 88.06% (R² Score)
- **Deployment Time**: < 3 minutes
- **Total Features**: 7 input features
- **Dataset Size**: 1,000 samples

---

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

1. **End-to-End ML Development**: From data collection to deployment
2. **Data Science**: EDA, feature engineering, model selection
3. **Software Engineering**: Modular code, error handling, logging
4. **Web Development**: Flask, HTML/CSS, user interfaces
5. **DevOps**: Git, GitHub, cloud deployment, CI/CD basics
6. **Best Practices**: Documentation, testing, version control

---

## ⭐ Show Your Support

If you found this project helpful, please consider:
- Giving it a ⭐ on GitHub
- Sharing it with others
- Contributing to improvements
- Providing feedback

---

<div align="center">

**Built with ❤️ using Python and Flask**

**[⬆ Back to Top](#-student-exam-performance-predictor)**

</div>
