---
name: sentiment-analysis-svm-nb
description: >
  Hướng dẫn toàn diện xây dựng dự án Phân tích Cảm xúc Khách hàng (Sentiment Analysis)
  dùng SVM và Naive Bayes để so sánh — dành cho bài tập môn học và báo cáo.
  Kích hoạt khi người dùng hỏi về sentiment analysis, phân tích cảm xúc, SVM, Naive Bayes,
  text classification, đánh giá sản phẩm, hoặc yêu cầu pipeline ML phân loại văn bản.
  Đọc skill này TRƯỚC KHI trả lời bất kỳ câu hỏi nào về các chủ đề trên.
---

# Sentiment Analysis — SVM vs Naive Bayes
## Skill dành cho bài tập môn học / báo cáo

---

## 🎯 Tổng quan dự án

**Bài toán:** Phân loại cảm xúc đánh giá khách hàng → Tích cực / Tiêu cực (/ Trung lập)

**Tại sao chọn Sentiment thay vì News Classification:**
- Output trực quan, giáo viên hiểu ngay không cần giải thích
- Gap SVM vs NB rõ ràng, có lý do phân tích sâu (negation, sarcasm)
- IMDB là benchmark cổ điển, dễ cite tài liệu tham khảo
- Demo thuyết phục: gõ 1 câu → ra nhãn ngay lập tức

```
Input  : "Sản phẩm rất tốt, giao hàng nhanh, sẽ mua lại!"
Output : POSITIVE  ✅

Input  : "Hàng kém chất lượng, không đúng mô tả, thất vọng."
Output : NEGATIVE  ✅

Input  : "not bad"   ← Naive Bayes dễ nhầm → SVM xử lý tốt hơn
Output : POSITIVE  (SVM đúng) / NEGATIVE (NB có thể sai)
```

---

## 📂 Cấu trúc dự án

```
sentiment-analysis/
├── SKILL.md                    ← file này — đọc trước
├── sentiment_svm_nb.py         ← pipeline hoàn chỉnh, chạy được ngay
├── notebooks/
│   └── sentiment_analysis.ipynb ← notebook phân tích từng bước
├── docs/
│   ├── theory.md               ← lý thuyết SVM + NB chi tiết
│   ├── report-outline.md       ← outline báo cáo môn học
│   └── real-data-results.md    ← kết quả thực nghiệm
├── data/
│   ├── raw/                    ← file CSV gốc sau khi tải
│   └── processed/              ← sau clean (tự sinh ra khi chạy code)
├── outputs/
│   ├── figures/                ← biểu đồ tự sinh
│   └── tables/                 ← bảng kết quả CSV
└── requirements.txt
```

**Khi nào đọc file nào:**
- Cần hiểu lý thuyết để viết báo cáo → đọc `docs/theory.md`
- Cần cấu trúc bài báo cáo → đọc `docs/report-outline.md`
- Chạy code ngay → xem `sentiment_svm_nb.py`
- Phân tích từng bước → mở `notebooks/sentiment_analysis.ipynb`

---

## 🗄️ CHỌN DATASET

| Option | Dataset | Lớp | Cỡ | Tải về | Khi nào dùng |
|--------|---------|-----|----|--------|--------------|
| **A ✅** | IMDB Movie Reviews | 2 (Pos/Neg) | 50K | Kaggle | Bắt đầu học, báo cáo đơn giản |
| **B ⭐** | Amazon Reviews | 3 (Neg/Neu/Pos) | 3.6M | Kaggle | Muốn multi-class, thực tế hơn |
| **C 🇻🇳** | UIT Vietnamese Feedback | 3 (Neg/Neu/Pos) | ~16K | HuggingFace | Tiếng Việt, phù hợp luận văn VN |

**Khuyên dùng cho báo cáo môn học: Option B (Amazon, 3 lớp)**
- Multi-class trình bày phong phú hơn binary
- Confusion matrix 3×3 đẹp hơn, nhiều điểm phân tích hơn
- Vẫn quen thuộc, nhiều tài liệu tham khảo

> **Ghi chú thực tế của repo này:** bản demo/notebook dùng **UCI Sentiment
> Labelled Sentences (2 lớp, 3.000 câu)** để nhẹ và chạy nhanh, không cần tải
> Kaggle. Amazon (3 lớp) và Vietnamese là các tuỳ chọn mở rộng — pipeline đã hỗ
> trợ sẵn qua `--source amazon` / `--source vn`.

```bash
# Tải IMDB (Option A):
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews

# Tải Amazon (Option B):
kaggle datasets download -d bittlingmayer/amazonreviews

# Tải Vietnamese (Option C):
pip install datasets
python -c "from datasets import load_dataset; ds = load_dataset('uitnlp/vietnamese_students_feedback'); ds['train'].to_csv('vn_feedback.csv')"
```

---

## ⚙️ PIPELINE 6 BƯỚC

### Bước 1 — Cài đặt môi trường

```bash
pip install scikit-learn pandas numpy matplotlib seaborn nltk kaggle
# Nếu dùng dataset Tiếng Việt thêm:
pip install underthesea datasets
```

---

### Bước 2 — Load & EDA (Exploratory Data Analysis)

**Checklist EDA — viết vào báo cáo:**
- [ ] Shape, dtypes, missing values → `df.info()`, `df.isnull().sum()`
- [ ] Phân phối nhãn → `df['sentiment'].value_counts()` — có mất cân bằng không?
- [ ] Độ dài text trung bình theo từng lớp
- [ ] Word cloud top 50 từ cho Positive vs Negative
- [ ] Top 20 từ phổ biến nhất (bar chart)

**Lưu ý cho báo cáo:** Nếu dataset mất cân bằng (imbalanced) → ghi chú và dùng `class_weight='balanced'` với SVM, `ComplementNB` thay vì `MultinomialNB`.

---

### Bước 3 — Tiền xử lý văn bản

**Flow:**
```
Raw text
  → Lowercase
  → Xóa HTML tags (<br />, <p>...)
  → Xóa URLs, số, ký tự đặc biệt
  → Tokenize
  → Xóa stopwords (NLTK English / underthesea cho Tiếng Việt)
  → Stemming (PorterStemmer) hoặc Lemmatization (WordNetLemmatizer)
  → Cleaned text
```

**Tiếng Anh (NLTK):**
```python
import re, nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer   = PorterStemmer()
stopwords = set(nltk.corpus.stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'<[^>]+>', ' ', text)        # HTML
    text = re.sub(r'http\S+', ' ', text)         # URLs
    text = re.sub(r'[^a-z\s]', ' ', text)        # Chỉ giữ chữ
    tokens = text.split()
    tokens = [stemmer.stem(w) for w in tokens
              if w not in stopwords and len(w) > 2]
    return ' '.join(tokens)
```

**Tiếng Việt (underthesea):**
```python
from underthesea import word_tokenize

def clean_text_vi(text):
    text = re.sub(r'http\S+', ' ', text)
    text = re.sub(r'[^\w\s]', ' ', text)
    tokens = word_tokenize(text.lower(), format='text').split()
    # Bỏ stopwords Tiếng Việt nếu có file
    return ' '.join(tokens)
```

---

### Bước 4 — Feature Engineering (Vectorization)

**Tóm tắt cho báo cáo:**

| Phương pháp | Ý nghĩa | Dùng với |
|---|---|---|
| `CountVectorizer` | Đếm tần suất thô của từ | Multinomial NB |
| `TfidfVectorizer` (1-gram) | TF × IDF — giảm trọng số từ phổ biến | SVM + NB |
| `TfidfVectorizer` (1+2-gram) | Bắt thêm cụm 2 từ liên tiếp | SVM (tốt nhất) |

**Công thức TF-IDF (viết vào lý thuyết báo cáo):**
```
TF(t,d)  = số lần từ t xuất hiện trong tài liệu d
IDF(t)   = log(N / df(t))  — N: tổng số tài liệu, df(t): số tài liệu chứa t
TF-IDF   = TF × IDF
sublinear_tf = True → dùng 1 + log(TF) thay vì TF — tốt hơn với text dài
```

**Config khuyên dùng:**
```python
TfidfVectorizer(
    ngram_range=(1, 2),     # unigram + bigram
    max_features=50_000,    # top 50K features
    sublinear_tf=True,      # log(tf)
    min_df=2,               # bỏ từ xuất hiện < 2 lần
)
```

---

### Bước 5 — Train & So sánh 4 mô hình

**4 mô hình đề xuất (2 NB variants + 2 SVM variants):**

| Mô hình | Class | Params | Đặc điểm |
|---|---|---|---|
| `MultinomialNB` | NB | `alpha=1.0` | Baseline nhanh nhất |
| `ComplementNB` | NB | `alpha=1.0` | Tốt hơn MNB khi dữ liệu lệch |
| `LinearSVC` | SVM | `C=1.0` | Nhanh, mạnh với text |
| `SVC(kernel='rbf')` | SVM | `C=1.0, gamma='scale'` | Chậm hơn, để so sánh kernel |

**Lưu ý quan trọng cho báo cáo:**
- Naive Bayes: KHÔNG cần scale dữ liệu
- SVM: KHÔNG cần scale khi dùng TF-IDF (sparse matrix, đã normalized)
- LinearSVC nhanh hơn SVC(rbf) ~10-100x với text → ưu tiên dùng LinearSVC

**Dùng Pipeline để tránh data leakage:**
```python
from sklearn.pipeline import Pipeline

pipeline = Pipeline([
    ('vec', TfidfVectorizer(...)),
    ('clf', LinearSVC(...)),
])
pipeline.fit(X_train, y_train)
```

---

### Bước 6 — Đánh giá & Trình bày kết quả

**Metrics bắt buộc trong báo cáo:**

| Metric | Ý nghĩa | Công thức |
|---|---|---|
| Accuracy | Tỷ lệ dự đoán đúng | TP+TN / Total |
| Precision | Trong các dự đoán Positive, bao nhiêu đúng | TP / (TP+FP) |
| Recall | Trong các Positive thực sự, bắt được bao nhiêu | TP / (TP+FN) |
| F1-Score | Trung bình điều hòa Precision & Recall | 2×P×R / (P+R) |
| Confusion Matrix | Ma trận nhầm lẫn — xem lớp nào sai ở đâu | Heatmap |

**Code đánh giá:**
```python
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

print(classification_report(y_test, y_pred, target_names=['Negative','Positive']))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Neg','Pos'], yticklabels=['Neg','Pos'])
```

**Biểu đồ cần có trong báo cáo:**
1. Bar chart phân phối nhãn (EDA)
2. Bảng so sánh Accuracy / F1 / Train time của 4 mô hình
3. Confusion Matrix của NB tốt nhất vs SVM tốt nhất (đặt cạnh nhau)
4. Top 15 từ quan trọng nhất của SVM (coef_ visualization)

---

## 🧠 PHÂN TÍCH SO SÁNH SVM vs NB — Nội dung viết báo cáo

### Tại sao SVM thường thắng NB trong Sentiment

**Naive Bayes — điểm yếu với Sentiment:**
```
Giả định độc lập (Naive assumption):
  P("not bad") = P("not") × P("bad")
  → NB xem "not" và "bad" độc lập nhau
  → Kết quả: "not bad" bị đánh giá tiêu cực (sai!)

Bigram (TF-IDF với ngram=2) giúp bắt được "not bad" như 1 feature
→ Cải thiện đáng kể, nhưng SVM vẫn xử lý tốt hơn về tổng thể
```

**SVM — lợi thế với Sentiment:**
```
Tìm hyperplane tối ưu trong không gian TF-IDF (sparse, high-dim)
Margin maximization → ít nhạy cảm với noise
LinearSVC: dùng hinge loss, rất hiệu quả với text classification
```

**Khi nào NB vẫn đủ tốt:**
- Prototype nhanh, dataset nhỏ
- Production cần tốc độ cực cao (realtime inference)
- Dataset sạch, từ ngữ rõ ràng (không có sarcasm, negation phức tạp)

---

## 📊 Kết quả kỳ vọng

**IMDB (2 lớp, 50K mẫu):**

| Mô hình | Accuracy | F1 | Train time |
|---|---|---|---|
| MultinomialNB | ~85-87% | ~85% | < 1s |
| ComplementNB | ~87-88% | ~87% | < 1s |
| LinearSVC | ~89-91% | ~90% | 5-10s |
| SVC (RBF) | ~89-91% | ~90% | 5-10 phút |

**Amazon (3 lớp, dùng 100K mẫu sample):**

| Mô hình | Accuracy | F1 (weighted) | F1 (macro) |
|---|---|---|---|
| MultinomialNB | ~82-84% | ~82% | ~75% |
| ComplementNB | ~84-86% | ~84% | ~77% |
| LinearSVC | ~88-90% | ~88% | ~82% |

*Lưu ý: F1 macro thấp hơn F1 weighted do class Neutral ít mẫu hơn*

---

## 🚀 Chạy nhanh

```bash
# Cài thư viện
pip install scikit-learn pandas numpy matplotlib seaborn nltk

# Chạy demo (5000 mẫu, không cần tải Kaggle):
python sentiment_svm_nb.py --demo

# Chạy full với IMDB:
python sentiment_svm_nb.py --source imdb --data imdb_dataset.csv

# Chạy với Amazon (3 lớp):
python sentiment_svm_nb.py --source amazon --data train.ft.txt

# Chạy với UCI (đã chuẩn bị sẵn):
python sentiment_svm_nb.py --source uci --data data/raw/uci_sentiment_reviews.csv
```

**Output files tự động sinh ra:**
```
01_eda.png                    → Biểu đồ EDA (phân phối nhãn + độ dài text)
02_model_comparison.png       → Bar chart so sánh 4 mô hình
03_confusion_matrices.png     → 4 confusion matrix cạnh nhau
04_top_features.png           → Top từ quan trọng nhất theo lớp
05_speed_comparison.png       → So sánh thời gian train
06_wordcloud.png              → Word cloud từ đặc trưng theo lớp
comparison_results.csv        → Bảng số liệu đầy đủ (copy vào báo cáo)
cross_validation_results.csv  → Kết quả cross-validation (5-fold)
hyperparameter_tuning.csv     → Kết quả tuning (GridSearchCV)
```

---

## 📚 Tài liệu tham khảo (dùng cite trong báo cáo)

- Scikit-learn docs: https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
- IMDB Dataset paper: Maas et al. (2011) — "Learning Word Vectors for Sentiment Analysis"
- SVM for Text Classification: Joachims (1998) — "Text Categorization with Support Vector Machines"
- NLTK Book: https://www.nltk.org/book/ (Chapter 6 — Learning to Classify Text)
- Underthesea (NLP Tiếng Việt): https://github.com/undertheseanlp/underthesea
