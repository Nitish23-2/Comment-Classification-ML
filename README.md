# Comment-Classification-ML
Multi-class comment classification.

# 🧠 Comment Category Prediction (Multi-Class NLP | Kaggle Project)

## 📌 Overview
This project focuses on solving a **multi-class text classification problem** where the goal is to predict how an online platform categorizes user-generated comments.

The dataset combines **textual, numerical, categorical, and temporal features**, making it a real-world structured + unstructured ML problem.

---

## 📊 Dataset Summary
- ~198,000 training samples
- 4 target classes (multi-class classification)
- Feature types:
  - 📝 Text: `comment`
  - 🔢 Numerical: `upvote`, `downvote`, `if_1`, `if_2`
  - 🏷️ Categorical: `race`, `religion`, `gender`, `disability`
  - ⏱️ Temporal: `created_date`

### ⚠️ Challenges
- Class imbalance across categories
- High missing values in categorical columns
- High-dimensional sparse text features
- Mixed data types (structured + unstructured)

---

## ⚙️ ML Pipeline

### 1️⃣ Data Preprocessing
- Filled missing text with empty strings
- Categorical missing values replaced with `"missing"`
- Log transformation (`log1p`) applied to skewed numerical features
- Datetime features converted into:
  - `hour`
  - `day_of_week`

---

### 2️⃣ Feature Engineering

#### 🔤 Text Features
- TF-IDF Vectorization:
  - Word-level (n-grams: 1 to 3)
  - Character-level (n-grams: 3 to 5)
- Captures both semantic and structural patterns

#### 🔢 Numerical Features
- `char_length`, `word_count`
- `total_votes`, `vote_ratio`
- `total_emoticons`

#### 🧠 Advanced Features
- Lexical diversity & mean word length
- Punctuation signals (`!`, `?`)
- Uppercase ratio & punctuation ratio
- Identity-based features:
  - `race_religion`
  - `race_gender`
- Hidden identity indicator feature

---

### 3️⃣ Feature Transformation
- StandardScaler → numerical features  
- OneHotEncoder → categorical features  
- Combined using sparse matrix (`hstack`)

---

### 4️⃣ Handling Class Imbalance
- Used `class_weight="balanced"`
- Applied SMOTE in selected experiments
- Optimized class-wise prediction thresholds

---

### 5️⃣ Models Implemented

#### ✅ Linear Models
- Logistic Regression  
- Linear SVM  

#### 🌲 Tree-Based Models
- XGBoost  
- LightGBM  

#### 🧠 Neural Network
- MLPClassifier  

---

### 6️⃣ Model Optimization
- Hyperparameter tuning (manual + randomized)
- Feature selection experiments (SVD tested)
- Ensemble strategies:
  - Weighted soft voting
  - Top model selection based on Macro F1

---

### 7️⃣ Final Model Strategy
- Selected top-performing models based on validation Macro F1
- Combined predictions using:
  - Weighted averaging  
  - Threshold optimization  

---

## 📈 Evaluation
- Metric used: **Macro F1 Score**
- Stratified train-validation split applied
- Focus on improving minority class performance

---

## 🏆 Key Results
- Linear models performed strongly on sparse TF-IDF features  
- Tree-based models improved after dimensionality reduction  
- Ensemble approach provided best overall performance  
- Threshold tuning significantly improved minority class predictions  

---

## 🧠 Key Learnings
- Feature engineering plays a critical role in NLP tasks  
- Linear models can outperform deep models on TF-IDF features  
- Handling class imbalance is essential for real-world datasets  
- Ensemble methods improve robustness and generalization  

---

## 🛠️ Tech Stack
- Python  
- Pandas, NumPy  
- Scikit-learn 
- Matplotlib, Seaborn  

---

## 🚀 Future Improvements
- Transformer-based models (BERT, DistilBERT)  
- Advanced NLP preprocessing (lemmatization, embeddings)  
- Better imbalance handling (focal loss)  
- Feature selection for dimensionality reduction  

---

## 👨‍💻 Author
Nitish Bhatt  
BS Data Science (IIT Madras) + B.Tech Mechanical Engineering (GBPUAT)
