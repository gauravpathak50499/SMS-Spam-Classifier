# SMS-Spam-Classifier
A machine learning project that classifies SMS messages as spam or ham (legitimate) using Natural Language Processing and a Multinomial Naive Bayes classifier.

# 📩 SMS Spam Classifier

A machine learning project that classifies SMS messages as **spam** or **ham** (legitimate) using Natural Language Processing and a **Multinomial Naive Bayes** classifier.

> Built with Python, NLTK, scikit-learn, and Streamlit.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Dataset](#-dataset)
- [Approach](#-approach)
- [Results](#-results)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Key Learnings](#-key-learnings)
- [Future Improvements](#-future-improvements)
- [Author](#-author)

---

## 🔍 Overview

Spam messages waste time and are often used for phishing and fraud. This project builds a text classifier that reads an SMS and predicts whether it is spam.

The notebook covers the full workflow: data cleaning, exploratory analysis, text preprocessing, feature extraction, model comparison across multiple algorithms, ensemble methods, and saving the final model for deployment.

---

## 📂 Dataset

- **Source:** [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) (UCI / Kaggle)
- **Size:** 5,572 labelled messages (before removing duplicates)
- **Columns:** `v1` (label: ham/spam) and `v2` (message text), renamed to `target` and `text`
- **Class balance:** heavily imbalanced, with ham far more common than spam. Because of this, accuracy alone is misleading, so precision, recall, and F1 are also reported.

**Cleaning steps**

1. Dropped three empty unnamed columns
2. Renamed columns and label-encoded the target (`ham = 0`, `spam = 1`)
3. Checked for null values
4. Removed duplicate messages

---

## 🛠 Approach

### 1. Exploratory Data Analysis
- Class distribution (pie chart)
- Engineered features: number of characters, words, and sentences per message
- Compared distributions for spam vs ham (histograms, pairplot, correlation heatmap). Spam messages tend to be longer.
- Word clouds and top-30 word frequency charts for each class

### 2. Text Preprocessing
Each message goes through a `transform_text()` function:

1. Lowercase the text
2. Tokenize with NLTK
3. Keep alphanumeric tokens only
4. Remove stopwords and punctuation
5. Apply Porter stemming

### 3. Feature Extraction
- Bag-of-words with `CountVectorizer` (TF-IDF is a natural alternative to compare)

### 4. Model Comparison
Trained and compared several classifiers on an 80/20 train-test split:

| Family | Models |
|---|---|
| Naive Bayes | Gaussian, Multinomial, Bernoulli |
| Linear / kernel | Logistic Regression, SVC |
| Tree-based | Decision Tree, Random Forest, Extra Trees |
| Other | K-Nearest Neighbors, AdaBoost, Bagging, Gradient Boosting, XGBoost |

I also tested **Voting** (soft) and **Stacking** ensembles built from RF, Extra Trees, XGBoost, and Logistic Regression.

### 5. Final Model
**Multinomial Naive Bayes** was selected because it:

- is designed for count-based text features,
- trains and predicts in milliseconds,
- gives a strong balance between precision and recall, and
- is simple to interpret and deploy.

---

## 📊 Results

| Metric | Multinomial NB |
|---|---|
| Accuracy | `0.9941972920696325` |
| Precision (spam) | `0.9925373134328358` |


**Confusion matrix**

|  | Predicted Ham | Predicted Spam |
|---|---|---|
| **Actual Ham** | `895` | `1` |
| **Actual Spam** | `5` | `133` |

**Why precision matters here:** a false positive means a real message is marked as spam, which is costly for users. A false negative means a spam message gets through. This project tracks both, not just accuracy.

---




### `requirements.txt`

```
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
xgboost
wordcloud
streamlit
```

---

## 💻 Usage

**Run the notebook**

```bash
jupyter notebook spamcl.ipynb
```

**Predict on a new message in Python**

```python
import pickle
from your_module import transform_text   # the preprocessing function from the notebook

cv = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open("model.pkl", "rb"))

msg = "Congratulations! You've won a free ticket. Reply WIN to claim now."
vector = cv.transform([transform_text(msg)])
print("Spam" if model.predict(vector)[0] == 1 else "Ham")
```



## 🧠 Key Learnings

- Why accuracy is a poor metric on imbalanced data, and how to use precision, recall, and F1 instead
- Building a text preprocessing pipeline with NLTK (tokenization, stopwords, stemming)
- Comparing many classical ML models fairly on the same features
- Why simple models like Naive Bayes remain strong baselines for text classification
- Ensemble methods: voting and stacking

---

## 🔮 Future Improvements

- Fit the vectorizer inside a scikit-learn `Pipeline` so it only sees training data (avoids data leakage)
- Use stratified k-fold cross-validation for more reliable model selection
- Try TF-IDF with n-grams and tune Naive Bayes smoothing (`alpha`) with `GridSearchCV`
- Add extra features (message length, digit count, presence of URLs or currency symbols)
- Experiment with transformer models such as DistilBERT
- Deploy the Streamlit app on Streamlit Community Cloud or Hugging Face Spaces

---

## 👤 Author

**Gaurav Pathak**
B.Tech in Computer Science & Business Systems, Techno India University, Kolkata


---

⭐ If you found this project useful, consider giving it a star!
