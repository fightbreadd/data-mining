
import pandas as pd
import re
import string
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from scipy.sparse import hstack

# 下载 NLTK 所需资源
nltk.download("stopwords")
nltk.download("wordnet")

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

# 读取数据
csv_path = "dataStore/Suicide_Detection.csv"
df = pd.read_csv(csv_path)

# 处理缺失值和重复值
df.dropna(subset=["text", "class"], inplace=True)
df.drop_duplicates(subset=["text"], inplace=True)
df["class"] = df["class"].map({"suicide": 1, "non-suicide": 0})

# 类别样本平衡（下采样）
min_count = df["class"].value_counts().min()
df = pd.concat([
    df[df["class"] == 0].sample(n=min_count, random_state=42),
    df[df["class"] == 1].sample(n=min_count, random_state=42)
]).sample(frac=1, random_state=42).reset_index(drop=True)

# 自定义停用词
stop_words = set(stopwords.words("english"))
custom_stop_words = stop_words.union({
    "im", "dont", "like", "just", "really", "people", "feel", "know", "want", "need", "think"
})

# 文本清洗函数（含 lemmatization）
def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"\d+", "", text)
    text = re.sub(r"\b[a-zA-Z]\b", "", text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stop_words]
    return " ".join(tokens)

df["text"] = df["text"].astype(str).apply(clean_text)

# 划分训练和测试集
X_train, X_test, y_train, y_test = train_test_split(df["text"], df["class"], test_size=0.2, random_state=42)

# TF-IDF 特征提取，包含 n-gram（1,2）
vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words=list(custom_stop_words),
    min_df=5,
    max_df=0.8,
    ngram_range=(1, 2)
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# 加载 GloVe 词向量（100维）
glove_embeddings = {}
with open("data/glove.6B.100d.txt", "r", encoding="utf8") as f:
    for line in f:
        parts = line.strip().split()
        word = parts[0]
        vec = np.array(parts[1:], dtype=float)
        glove_embeddings[word] = vec

def get_glove_vector(text):
    words = text.split()
    vectors = [glove_embeddings[w] for w in words if w in glove_embeddings]
    return np.mean(vectors, axis=0) if vectors else np.zeros(100)

X_train_glove = np.vstack(X_train.apply(get_glove_vector))
X_test_glove = np.vstack(X_test.apply(get_glove_vector))

# 合并 TF-IDF 和 GloVe 向量
X_train_combined = hstack([X_train_tfidf, X_train_glove])
X_test_combined = hstack([X_test_tfidf, X_test_glove])

# 可视化 TF-IDF 关键词
def plot_tfidf_top_words(vectorizer, X_train_tfidf, top_n=20):
    feature_array = np.array(vectorizer.get_feature_names_out())
    tfidf_mean = np.mean(X_train_tfidf.toarray(), axis=0)
    tfidf_sorted_indices = np.argsort(tfidf_mean)[::-1]
    top_features = feature_array[tfidf_sorted_indices][:top_n]
    top_values = tfidf_mean[tfidf_sorted_indices][:top_n]
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_values[::-1], y=top_features[::-1])
    plt.xlabel("TF-IDF Score")
    plt.ylabel("Words")
    plt.title(f"Top {top_n} Important Words in Suicide Detection (TF-IDF)")
    plt.show()

plot_tfidf_top_words(vectorizer, X_train_tfidf)

# SVM 模型训练
svm_model = LinearSVC(C=0.5, max_iter=2000)
start_time = time.time()
svm_model.fit(X_train_combined, y_train)
end_time = time.time()
print(f"Training completed in {end_time - start_time:.2f} seconds")

# 模型评估
y_pred = svm_model.predict(X_test_combined)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.4f}")
print("Classification Report:\n", classification_report(y_test, y_pred))

# 降维可视化
def plot_svm_tsne(X, y):
    print("Running PCA...")
    pca = PCA(n_components=50, random_state=42)
    X_pca = pca.fit_transform(X.toarray())
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=500, random_state=42)
    X_tsne = tsne.fit_transform(X_pca)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='coolwarm', alpha=0.5)
    plt.legend(handles=scatter.legend_elements()[0], labels=["Non-Suicide (0)", "Suicide (1)"])
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.title("SVM Classification Result After PCA + t-SNE")
    plt.show()

plot_svm_tsne(X_test_combined, y_test)
