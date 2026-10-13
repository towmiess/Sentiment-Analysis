# Outline Báo cáo Môn học
## Phân tích Cảm xúc Khách hàng sử dụng SVM và Naive Bayes

> **Dữ liệu thực tế của dự án:** UCI Sentiment Labelled Sentences (2 lớp,
> 3.000 câu, cân bằng). Outline dưới đây là mẫu chung — có thể thay bằng Amazon
> (3 lớp) hoặc UIT-VSFC (tiếng Việt) tuỳ yêu cầu bài tập.

---

## Cấu trúc báo cáo (gợi ý ~15-20 trang)

---

### I. GIỚI THIỆU (1-2 trang)

**1.1 Đặt vấn đề**
- Sự bùng nổ của review online (Amazon, Shopee, Google Maps...)
- Doanh nghiệp cần tự động phân tích hàng triệu đánh giá
- Thách thức: ngôn ngữ tự nhiên phức tạp, sarcasm, negation

**1.2 Mục tiêu**
- Xây dựng pipeline phân loại sentiment tự động
- So sánh hiệu năng giữa SVM và Naive Bayes
- Đánh giá ưu/nhược điểm của từng phương pháp

**1.3 Phạm vi**
- Dataset: [IMDB / Amazon / Vietnamese Feedback] — ghi rõ option đã chọn
- Ngôn ngữ: Tiếng Anh / Tiếng Việt
- Nhãn phân loại: Positive / Negative [/ Neutral nếu 3 lớp]

---

### II. CƠ SỞ LÝ THUYẾT (3-4 trang)

**2.1 Tổng quan Text Classification**
- Bag of Words model
- TF-IDF — công thức và ý nghĩa
- N-gram và lý do dùng bigram

**2.2 Naive Bayes**
- Định lý Bayes (viết công thức)
- Giả định "Naive" và ý nghĩa
- Multinomial NB và Complement NB
- Laplace Smoothing
- Ưu / Nhược điểm

**2.3 Support Vector Machine**
- Hyperplane và Margin (vẽ hình minh họa)
- Support Vectors
- Soft Margin và tham số C
- Linear Kernel với text data
- LinearSVC vs SVC
- Ưu / Nhược điểm

**2.4 Metrics đánh giá**
- Accuracy, Precision, Recall, F1-Score (viết công thức, giải thích)
- Confusion Matrix
- Tại sao F1 quan trọng hơn Accuracy với imbalanced data

---

### III. DỮ LIỆU (2-3 trang)

**3.1 Mô tả dataset**
- Nguồn gốc: [IMDB Kaggle / Amazon / UIT-VSFC]
- Số lượng mẫu, số lớp, ngôn ngữ
- Phân phối nhãn (kèm biểu đồ bar chart)

**3.2 Phân tích khám phá (EDA)**
- Bảng thống kê mô tả (mean length, max, min...)
- Biểu đồ phân phối độ dài văn bản theo lớp
- Word cloud của từng lớp sentiment
- Top 20 từ phổ biến nhất

**3.3 Tiền xử lý văn bản**
- Các bước xử lý (liệt kê flow: lowercase → remove HTML → tokenize...)
- Ví dụ trước/sau xử lý (bảng 2-3 ví dụ)
- Lý do chọn từng bước

---

### IV. PHƯƠNG PHÁP THỰC HIỆN (2-3 trang)

**4.1 Kiến trúc pipeline**

```
Raw Text
  → Tiền xử lý (Bước 3)
  → TF-IDF Vectorization
  → [MultinomialNB | ComplementNB | LinearSVC | SVC(RBF)]
  → Nhãn dự đoán
  → Đánh giá (Accuracy, F1, Confusion Matrix)
```

**4.2 Cấu hình thực nghiệm**
- Chia train/test: 80/20, stratified sampling
- TF-IDF params: ngram_range=(1,2), max_features=50000, sublinear_tf=True
- Các mô hình và hyperparameters (bảng)

**4.3 Công cụ và thư viện**
- Python 3.x, scikit-learn, pandas, numpy, matplotlib, seaborn, NLTK

---

### V. KẾT QUẢ VÀ ĐÁNH GIÁ (4-5 trang)

**5.1 Kết quả EDA**
- Nhận xét về phân phối dữ liệu
- Các từ đặc trưng của từng lớp

**5.2 Bảng so sánh hiệu năng**

| Mô hình | Accuracy | Precision | Recall | F1 | Train time |
|---|---|---|---|---|---|
| MultinomialNB | ... | ... | ... | ... | ... |
| ComplementNB | ... | ... | ... | ... | ... |
| LinearSVC | ... | ... | ... | ... | ... |
| SVC (RBF) | ... | ... | ... | ... | ... |

**5.3 Phân tích Confusion Matrix**
- Vẽ heatmap confusion matrix của NB tốt nhất và SVM tốt nhất
- Nhận xét: lớp nào bị nhầm nhiều? Tại sao?

**5.4 Phân tích đặc trưng quan trọng**
- Top features của LinearSVC (coef_ visualization)
- So sánh với top features của NB
- Nhận xét: các từ này có ý nghĩa ngữ nghĩa không?

**5.5 Demo inference**
- Chạy thử trên 5-10 câu tự đặt ra
- Phân tích trường hợp mô hình sai: tại sao NB sai, SVM đúng?
- Ví dụ điển hình: negation ("not good"), sarcasm ("Oh great, another bug")

---

### VI. THẢO LUẬN (1-2 trang)

**6.1 So sánh SVM vs Naive Bayes**
- Khi nào nên dùng NB (prototype nhanh, data nhỏ, text rất ngắn)
- Khi nào nên dùng SVM (production, accuracy quan trọng)
- Trade-off: accuracy vs speed vs interpretability

**6.2 Hạn chế của nghiên cứu**
- Không xử lý được emoji, slang, viết tắt
- Chưa xử lý sarcasm phức tạp
- Chỉ dùng bag-of-words, chưa có context (so sánh với BERT)

**6.3 Hướng phát triển**
- Dùng word embeddings (Word2Vec, GloVe) thay TF-IDF
- Fine-tune BERT/PhoBERT cho Tiếng Việt
- Thêm feature: emoticons, capitalization, punctuation pattern

---

### VII. KẾT LUẬN (0.5-1 trang)

- Tóm tắt kết quả đạt được
- Mô hình nào tốt nhất, lý do
- Bài học rút ra từ thực nghiệm
- Ý nghĩa thực tiễn của bài toán

---

### TÀI LIỆU THAM KHẢO

1. Maas, A. L., et al. (2011). Learning Word Vectors for Sentiment Analysis. ACL 2011.
2. Joachims, T. (1998). Text Categorization with Support Vector Machines. ECML 1998.
3. McCallum, A., & Nigam, K. (1998). A Comparison of Event Models for Naive Bayes Text Classification. AAAI Workshop.
4. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. JMLR, 12, 2825-2830.
5. [Dataset citation theo dataset bạn dùng]
6. Bird, S., Klein, E., & Loper, E. (2009). Natural Language Processing with Python. O'Reilly.

---

## ⏱️ Ước tính thời gian làm

| Phần | Thời gian |
|---|---|
| Setup môi trường + tải data | 1-2 giờ |
| Chạy code pipeline | 1-2 giờ |
| Phân tích kết quả + vẽ biểu đồ | 2-3 giờ |
| Viết báo cáo | 4-6 giờ |
| **Tổng** | **~1-2 ngày** |
