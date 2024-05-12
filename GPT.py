import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import GPT2Tokenizer, GPT2Model
import torch

# GPT modelini yükle
model_name = "ytu-ce-cosmos/turkish-gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2Model.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

def gpt_encode(texts):
    vectors = []
    for i, text in enumerate(texts):
        encoded_input = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            output = model(**encoded_input)
        vectors.append(output.last_hidden_state[:, 0, :].squeeze().numpy())
        if (i+1) % 100 == 0 or i == len(texts) - 1:
            print(f"Processing text {i+1}/{len(texts)}")
    return vectors

# Veri setini yükle
df = pd.read_csv("dataset.csv")

# 'insan' ve 'makine' olarak iki farklı sütun oluştur ve bunları birleştir
insan_df = df[['insan cevabı']].rename(columns={'insan cevabı': 'text'})
insan_df['label'] = 'insan'
makine_df = df[['makine cevabı']].rename(columns={'makine cevabı': 'text'})
makine_df['label'] = 'makine'
combined_df = pd.concat([insan_df, makine_df], ignore_index=True)

# Veriyi karıştır ve eğitim setine ayır
combined_df = combined_df.sample(frac=1).reset_index(drop=True)
X = combined_df['text']
y = combined_df['label']

# Metinleri GPT ile vektörleştir
X_encoded = gpt_encode(X)

# Sınıflandırıcılar
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Naive Bayes": GaussianNB(),
    "SVM": SVC(random_state=42),
    "SVM with Scaling": make_pipeline(StandardScaler(with_mean=False), SVC(random_state=42))
}

# Sonuçları değerlendir
results = {}
for c_name, classifier in classifiers.items():
    scores = cross_val_score(classifier, X_encoded, y, cv=5)
    results[c_name] = scores.mean()

# Sonuçları yazdır
for clf_name, score in results.items():
    print(f"{clf_name}: {score:.4f}")
