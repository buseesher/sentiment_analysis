# 🛍️ Amazon Review Sentiment Analysis & Modeling

This project performs sentiment analysis and machine learning-based sentiment prediction using customer reviews from Amazon product data. It includes text preprocessing, visualization, sentiment labeling using NLTK, and modeling using Logistic Regression and Random Forest.

---

## 📑 Dataset Description

The dataset (`amazon.xlsx`) contains the following columns:

- **Review**: Full user comment
- **Title**: Short summary of the review
- **Helpful**: Number of users who found the review helpful
- **Star**: Rating given to the product (1–5 stars)

---

## 🧠 Tasks Covered

### 🔹 Task 1: Text Preprocessing

- Convert text to lowercase
- Remove punctuation and numbers
- Filter English stopwords
- Remove rare words (bottom 1000 by frequency)
- Lemmatize all tokens

### 🔹 Task 2: Text Visualization

- **Barplot** of most frequent words (`tf > 500`)
- **WordCloud** showing top 100 words by frequency

### 🔹 Task 3: Sentiment Analysis

- Uses **VADER** (SentimentIntensityAnalyzer) from NLTK
- Calculates polarity scores (compound)
- Labels each review as:
  - `"pos"` if compound score > 0
  - `"neg"` otherwise
- Adds `Sentiment_Label` column to dataset

### 🔹 Task 4: Machine Learning Preparation

- Splits data into **train/test**
- Applies **TF-IDF vectorization** to convert text to numeric features

### 🔹 Task 5: Modeling with Logistic Regression

- Trains `LogisticRegression` model
- Evaluates with:
  - `classification_report`
  - `cross_val_score` (5-fold)
- Allows prediction for a **randomly sampled review**

### 🔹 Task 6: Modeling with Random Forest

- Trains a `RandomForestClassifier`
- Compares accuracy with logistic regression model

---

## 📦 Requirements

Install required Python packages:

```bash
 pip install pandas matplotlib wordcloud nltk textblob scikit-learn openpyxl
  ```
Also download required NLTK corpora:
```bash
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('vader_lexicon')
  ```
---

## ▶️ How to Run

1.Place amazon.xlsx in the project directory.

2.Run the main script in any Python environment.

3.Review sentiment classification results and model performance.

---

## 📊 Sample Output

```bash
Review:  the product quality is amazing and delivery was fast  
Prediction: ['pos']
  ```
