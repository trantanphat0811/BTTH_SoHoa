import numpy as np
import pandas as pd
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import random
from nltk.corpus import wordnet
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')

# Load dataset
print(" Đang tải dữ liệu...")
df = pd.read_csv('DoAn/mail_data.csv', encoding='utf-8')
df.fillna('', inplace=True)
print(f" Số dòng dữ liệu: {len(df)}")

df['Category'] = df['Category'].map({'spam': 0, 'ham': 1})


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)  # Xóa số
    text = re.sub(r'[^\w\s@.]', '', text)  # Giữ lại @ và .
    text = text.strip()
    return text


df['Message'] = df['Message'].apply(preprocess_text)

# Synonym Replacement


def synonym_replacement(text):
    words = text.split()
    new_words = []
    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            new_word = synonyms[0].lemmas()[0].name()
            new_words.append(new_word)
        else:
            new_words.append(word)
    return ' '.join(new_words)


# Data Augmentation for spam
df_spam = df[df['Category'] == 0]
df_spam_aug = df_spam.copy()
df_spam_aug['Message'] = df_spam_aug['Message'].apply(synonym_replacement)
df_spam = pd.concat([df_spam, df_spam_aug])

# Under-Sampling for ham
count_spam = len(df_spam)
df_ham = df[df['Category'] == 1].sample(n=count_spam, random_state=42)

df = pd.concat([df_spam, df_ham]).sample(frac=1, random_state=42)

# Split data
X = df['Message']
Y = df['Category']
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42, stratify=Y)

vectorizer = TfidfVectorizer(min_df=1, lowercase=True, stop_words='english')
X_train_features = vectorizer.fit_transform(X_train)
X_test_features = vectorizer.transform(X_test)

Y_train = Y_train.astype(int)
Y_test = Y_test.astype(int)

print(" Đang huấn luyện mô hình...")
model = LogisticRegression(C=1.0, max_iter=500, solver='liblinear')
model.fit(X_train_features, Y_train)

train_accuracy = accuracy_score(Y_train, model.predict(X_train_features))
test_accuracy = accuracy_score(Y_test, model.predict(X_test_features))

print(f' Training Accuracy: {train_accuracy:.4f}')
print(f' Test Accuracy: {test_accuracy:.4f}')

with open('spam_classifier.pkl', 'wb') as f:
    pickle.dump((vectorizer, model), f)

print(" Mô hình đã được lưu thành công: spam_classifier.pkl")
