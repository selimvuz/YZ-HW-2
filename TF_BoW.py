import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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

# Metin temsil yöntemleri
vectorizers = {
    'TF-IDF': TfidfVectorizer(max_features=5000),
    'Count Vectorizer': CountVectorizer(max_features=5000),
}

# Sınıflandırıcılar
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Multinomial Naive Bayes": GaussianNB(),
    "SVM": SVC(random_state=42),
    "SVM with Scaling": make_pipeline(StandardScaler(with_mean=False), SVC(random_state=42))
}

# Sonuçları değerlendir
all_results = {}
for v_name, vectorizer in vectorizers.items():
    X_transformed = vectorizer.fit_transform(X)
    results = {}
    for c_name, classifier in classifiers.items():
        X_transformed_dense = X_transformed.toarray()
        scores = cross_val_score(classifier, X_transformed_dense, y, cv=5)
        results[c_name] = scores.mean()
    all_results[v_name] = results

# Sonuçları konsola yazdır
for vect_name, scores in all_results.items():
    print(f"---{vect_name}---")
    for clf_name, score in scores.items():
        print(f"{clf_name}: {score:.4f}")