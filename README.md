# Phân tích Cảm xúc Khách hàng: SVM vs Naive Bayes

> **Sentiment Analysis — Support Vector Machine vs Naive Bayes**
> Bài tập môn học / báo cáo: so sánh hai họ mô hình học máy cổ điển trên bài toán
> phân loại cảm xúc đánh giá khách hàng (tích cực / tiêu cực [/ trung lập]).
>
> 📘 **Dành cho người dùng / khách hàng:** xem cẩm nang cài đặt – chạy – thực nghiệm
> chi tiết tại **[`HUONG_DAN.md`](HUONG_DAN.md)**.

---

## 🎯 Tổng quan

Pipeline hoàn chỉnh từ **tải dữ liệu → EDA → tiền xử lý → vector hóa TF-IDF →
huấn luyện 4 mô hình → đánh giá (Accuracy/F1/Confusion Matrix/Cross-Validation/
Tuning) → trực quan hóa → demo inference**. Hai họ mô hình được so sánh:

| Họ | Mô hình | Đặc điểm |
|---|---|---|
| **Naive Bayes** | `MultinomialNB`, `ComplementNB` | Nhanh, đơn giản, yếu với phủ định |
| **SVM** | `LinearSVC`, `SVC (RBF)` | Margin tối ưu, xử lý tốt high-dim sparse |

**Dataset:** UCI Sentiment Labelled Sentences (3.000 câu review IMDb/Amazon/Yelp,
cân bằng 1.500 tích cực / 1.500 tiêu cực) — xem `data/raw/README.md`.
Hỗ trợ thêm IMDB (2 lớp), Amazon (3 lớp) và UIT Vietnamese Students' Feedback (3 lớp, tiếng Việt).

## 📊 Kết quả tóm tắt (UCI, 80/20 split, `random_state=42`)

| Mô hình | Họ | Accuracy | F1 (weighted) |
|---|---:|---:|---:|
| **SVC (RBF)** | SVM | **0.8283** | **0.8283** |
| ComplementNB | Naive Bayes | 0.8233 | 0.8233 |
| LinearSVC | SVM | 0.8233 | 0.8233 |
| MultinomialNB | Naive Bayes | 0.8050 | 0.8050 |

> Chi tiết: accuracy/F1 theo lớp, ma trận nhầm lẫn, cross-validation (5-fold),
> hyperparameter tuning — xem `docs/theory.md` (mục 8–9) và `docs/real-data-results.md`.
> *Thời gian train/predict là phép đo cục bộ và dao động theo máy.*

## 📂 Cấu trúc dự án

```
sentiment-analysis/
├── sentiment_svm_nb.py            # pipeline CLI (imdb/amazon/uci/vn + demo)
├── notebooks/sentiment_analysis.ipynb   # notebook phân tích từng bước
├── docs/
│   ├── theory.md                  # lý thuyết SVM + NB + kết quả chi tiết
│   ├── report-outline.md          # outline báo cáo môn học
│   └── real-data-results.md       # kết quả thực nghiệm (tiếng Anh)
├── data/raw/                      # dataset gốc (CSV + nguồn tải)
├── outputs/
│   ├── figures/                   # EDA, so sánh, confusion, features, word cloud
│   └── tables/                    # comparison, cross-validation, tuning CSV
├── requirements.txt
└── SKILL.md                       # hướng dẫn toàn diện (đọc trước)
```

## ⚙️ Cài đặt

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

> Dùng dataset tiếng Việt cần thêm `underthesea` và `datasets` (đã có trong
> `requirements.txt`).

## 🚀 Chạy nhanh

```powershell
# Demo 5.000 mẫu tổng hợp (không cần tải dữ liệu)
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --demo

# UCI Sentiment Labelled Sentences (đã chuẩn bị sẵn)
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --source uci --data .\data\raw\uci_sentiment_reviews.csv

# IMDB (tải từ Kaggle, CSV có cột 'review' + 'sentiment')
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --source imdb --data .\data\raw\imdb_dataset.csv

# Amazon Reviews (file train.ft.txt, gộp thành 2 lớp pos/neg)
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --source amazon --data .\data\raw\amazon_reviews.ft.txt

# Tiếng Việt (UIT-VSFC, 3 lớp: negative/neutral/positive)
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --source vn --data .\data\raw\vn_feedback.csv
```

Tham số hữu ích: `--sample 10000` (giới hạn số mẫu), `--output-dir outputs`
(thư mục xuất kết quả).

### Notebook

Mở `notebooks/sentiment_analysis.ipynb` bằng VS Code hoặc Jupyter và chạy các cell
từ trên xuống dưới.

## 📁 Đầu ra

Pipeline tự sinh các file trong `outputs/`:

- `figures/`: `01_eda`, `02_model_comparison`, `03_confusion_matrices`,
  `04_top_features`, `05_speed_comparison`, `06_wordcloud`.
- `tables/`: `comparison_results.csv`, `cross_validation_results.csv`,
  `hyperparameter_tuning.csv`.

## 📚 Tài liệu & trích dẫn

- `HUONG_DAN.md` — cẩm nang cài đặt, chạy & thực nghiệm cho người dùng (đọc trước).
- `SKILL.md` — hướng dẫn toàn diện (đọc trước).
- `docs/theory.md` — cơ sở lý thuyết SVM & Naive Bayes.
- `docs/report-outline.md` — cấu trúc báo cáo môn học gợi ý.
- `docs/real-data-results.md` — kết quả thực nghiệm và giới hạn diễn giải.

**Dataset:** Kotzias, D., Denil, M., de Freitas, N., & Smyth, P. (2015).
*From Group to Individual Labels Using Deep Features*. KDD 2015.
UCI Machine Learning Repository, DOI: `10.24432/C57604`.

## 📄 License

MIT — xem `LICENSE`.

