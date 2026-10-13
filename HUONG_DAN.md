# 📘 Hướng dẫn sử dụng & thực nghiệm

> **Dự án: Phân tích Cảm xúc Khách hàng — SVM vs Naive Bayes**
> Tài liệu dành cho người dùng cuối (khách hàng / giảng viên / người đánh giá):
> cài đặt, chạy lại thí nghiệm và tự thực nghiệm trên dữ liệu riêng.

---

## Mục lục

1. [Giới thiệu dự án](#1-giới-thiệu-dự-án)
2. [Yêu cầu hệ thống](#2-yêu-cầu-hệ-thống)
3. [Cấu trúc thư mục](#3-cấu-trúc-thư-mục)
4. [Cài đặt môi trường](#4-cài-đặt-môi-trường)
5. [Chạy nhanh lần đầu (Demo)](#5-chạy-nhanh-lần-đầu-demo)
6. [Chạy trên dữ liệu thật](#6-chạy-trên-dữ-liệu-thật)
7. [Chạy Notebook phân tích từng bước](#7-chạy-notebook-phân-tích-từng-bước)
8. [Đọc & diễn giải kết quả](#8-đọc--diễn-giải-kết-quả)
9. [Hướng dẫn thực nghiệm (tùy chỉnh)](#9-hướng-dẫn-thực-nghiệm-tùy-chỉnh)
10. [Khắc phục lỗi thường gặp](#10-khắc-phục-lỗi-thường-gặp)
11. [Câu hỏi thường gặp (FAQ)](#11-câu-hỏi-thường-gặp-faq)
12. [Trích dẫn & liên hệ](#12-trích-dẫn--liên-hệ)

---

## 1. Giới thiệu dự án

Dự án so sánh **hai họ mô hình học máy cổ điển** trên bài toán **phân loại cảm xúc**
văn bản (review khách hàng → tích cực / tiêu cực [/ trung lập]):

| Họ | Mô hình | Đặc điểm |
|---|---|---|
| **Naive Bayes** | `MultinomialNB`, `ComplementNB` | Nhanh, đơn giản, dựa trên định lý Bayes |
| **SVM** | `LinearSVC`, `SVC (RBF)` | Tìm siêu phẳng biên lớn nhất, mạnh với văn bản |

Pipeline đầy đủ: **tải dữ liệu → EDA → tiền xử lý → vector hóa TF-IDF → huấn luyện
4 mô hình → đánh giá (Accuracy / F1 / Confusion Matrix / Cross-Validation / Tuning) →
biểu đồ → demo dự đoán**.

Bạn có thể:
- **Chạy lại toàn bộ thí nghiệm** chỉ bằng **một dòng lệnh**.
- **Đọc kết quả có sẵn** (biểu đồ + bảng số) ngay, không cần cài gì.
- **Thực nghiệm** trên dữ liệu riêng hoặc chỉnh tham số mô hình.

Cơ sở lý thuyết chi tiết: xem `docs/theory.md`.

---

## 2. Yêu cầu hệ thống

| Thành phần | Yêu cầu |
|---|---|
| Hệ điều hành | Windows 10/11, macOS, hoặc Linux |
| Python | **3.9 – 3.12** (khuyên dùng 3.10 hoặc 3.11) |
| RAM | ≥ 4 GB (khuyến nghị 8 GB) |
| Ổ đĩa trống | ~1 GB (môi trường ảo + dữ liệu) |
| GPU | Không cần (mô hình chạy trên CPU) |
| Internet | Cần cho lần cài thư viện đầu và khi tải thêm dataset |

Kiểm tra Python đã cài:

```powershell
python --version
```

---

## 3. Cấu trúc thư mục

```
sentiment-analysis-v1.0/
├── README.md                      ← giới thiệu ngắn gọn
├── HUONG_DAN.md                   ← TÀI LIỆU NÀY (đọc trước)
├── sentiment_svm_nb.py            ← pipeline CLI, chạy 1 dòng lệnh
├── notebooks/
│   └── sentiment_analysis.ipynb   ← phân tích từng bước (Jupyter)
├── docs/
│   ├── theory.md                  ← lý thuyết SVM + NB chi tiết
│   ├── report-outline.md          ← dàn ý báo cáo
│   └── real-data-results.md       ← kết quả thực nghiệm (tiếng Anh)
├── data/
│   └── raw/                       ← dữ liệu gốc (đã có sẵn 2 dataset)
│       ├── uci_sentiment_reviews.csv   (UCI, 3.000 câu, 2 lớp)
│       └── vn_feedback.csv             (UIT-VSFC, ~16K câu, 3 lớp, tiếng Việt)
├── outputs/                       ← kết quả tiếng Anh (biểu đồ + bảng CSV)
├── outputs_vn/                    ← kết quả tiếng Việt
├── scripts/
│   └── prepare_vn_dataset.py      ← tải lại dataset tiếng Việt (tùy chọn)
├── requirements.txt               ← danh sách thư viện cần cài
└── LICENSE                        ← MIT
```

- `outputs/` — kết quả trên **UCI (tiếng Anh, 2 lớp)**.
- `outputs_vn/` — kết quả trên **UIT-VSFC (tiếng Việt, 3 lớp)**.
- Hai thư mục này **đã có sẵn kết quả** — bạn xem được ngay mà không cần chạy lại.

---

## 4. Cài đặt môi trường

### 4.1 Tạo môi trường ảo (khuyến nghị)

**Windows (PowerShell):**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux (bash):**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4.2 Cài thư viện

```powershell
pip install -r requirements.txt
```

> Việc cài có thể mất vài phút (pandas, scikit-learn, matplotlib, seaborn, nltk,
> wordcloud, underthesea…). Đảm bảo máy có kết nối Internet.

### 4.3 Kiểm tra cài đặt thành công

```powershell
.\.venv\Scripts\python.exe -c "import sklearn, pandas, numpy, matplotlib, seaborn, nltk; print('OK')"
```

Nếu in ra `OK` là môi trường đã sẵn sàng.

## 5. Chạy nhanh lần đầu (Demo)

Lệnh demo dùng **5.000 mẫu tổng hợp**, không cần tải bất kỳ dữ liệu nào — dùng để
kiểm tra môi trường chạy ổn:

```powershell
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --demo
```

Chương trình sẽ in ra màn hình tiến trình: nạp dữ liệu → EDA → huấn luyện 4 mô hình →
bảng kết quả so sánh (Accuracy / F1 / thời gian) → cross-validation → tuning → sinh biểu
đồ → demo dự đoán vài câu ví dụ. Kết thúc, các file kết quả nằm trong thư mục `outputs/`.

> ⚠️ Chạy `--demo` sẽ **ghi đè** kết quả có sẵn trong `outputs/`. Muốn giữ kết quả gốc,
> xem mục 9.2 để xuất sang thư mục khác.

---

## 6. Chạy trên dữ liệu thật

Bảng tóm tắt các lệnh (chạy từ thư mục gốc dự án):

| Bộ dữ liệu | Lớp | Dữ liệu | Lệnh |
|---|---|---|---|
| **UCI (đã có sẵn)** | 2 | 3.000 câu | `--source uci --data .\data\raw\uci_sentiment_reviews.csv` |
| **UIT-VSFC tiếng Việt (đã có sẵn)** | 3 | ~16K câu | `--source vn --data .\data\raw\vn_feedback.csv` |
| IMDB (tự tải) | 2 | 50K | `--source imdb --data .\data\raw\imdb_dataset.csv` |
| Amazon (tự tải) | 2 | lớn | `--source amazon --data .\data\raw\amazon_reviews.ft.txt` |

### 6.1 UCI Sentiment Labelled Sentences (đã có sẵn)

```powershell
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --source uci --data .\data\raw\uci_sentiment_reviews.csv
```

### 6.2 Tiếng Việt — UIT-VSFC (đã có sẵn, 3 lớp)

```powershell
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --source vn --data .\data\raw\vn_feedback.csv
```

> Dataset tiếng Việt cần thư viện `underthesea` (đã có trong `requirements.txt`).
> Nếu muốn tải lại từ nguồn: `python scripts\prepare_vn_dataset.py`.

### 6.3 IMDB (tự tải từ Kaggle)

Tải file CSV (cột `review` + `sentiment`), đặt vào `data/raw/`:

```powershell
kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
```

Giải nén rồi chạy:

```powershell
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --source imdb --data .\data\raw\imdb_dataset.csv
```

### 6.4 Amazon Reviews (tự tải từ Kaggle, dạng `.ft.txt`)

```powershell
kaggle datasets download -d bittlingmayer/amazonreviews
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --source amazon --data .\data\raw\amazon_reviews.ft.txt
```

> Amazon rất lớn (hàng triệu dòng). Nên giới hạn số mẫu: thêm `--sample 50000`.

### 6.5 Các tham số dòng lệnh

| Tham số | Ý nghĩa | Ví dụ |
|---|---|---|
| `--source` | Bộ nạp dữ liệu: `imdb` / `amazon` / `uci` / `vn` | `--source vn` |
| `--data` | Đường dẫn file dữ liệu | `--data .\data\raw\vn_feedback.csv` |
| `--demo` | Chạy dữ liệu mẫu, không cần file | `--demo` |
| `--sample N` | Giới hạn số mẫu (chạy nhanh hơn) | `--sample 10000` |
| `--output-dir` | Thư mục xuất kết quả | `--output-dir outputs_test` |

---

## 7. Chạy Notebook phân tích từng bước

File `notebooks/sentiment_analysis.ipynb` tái hiện từng bước phân tích (EDA, tiền xử lý,
TF-IDF, huấn luyện, đánh giá, biểu đồ, tuning) dưới dạng các cell riêng — phù hợp để
**học** và **điều chỉnh từng đoạn**.

Cách mở:

- **VS Code**: mở thư mục dự án → mở file notebook → chọn kernel `.venv` → chạy từng cell.
- **Jupyter**: chạy lệnh sau rồi mở trong trình duyệt:

```powershell
.\.venv\Scripts\python.exe -m jupyter notebook notebooks\sentiment_analysis.ipynb
```

Chạy các cell **từ trên xuống dưới** (cell sau dùng biến của cell trước).

---

## 8. Đọc & diễn giải kết quả

### 8.1 Kết quả định lượng (số liệu chính)

**UCI — tiếng Anh, 2 lớp (80/20 split, `random_state=42`):**

| Mô hình | Họ | Accuracy | F1 (weighted) |
|---|---:|---:|---:|
| **SVC (RBF)** | SVM | **0.8283** | **0.8283** |
| ComplementNB | Naive Bayes | 0.8233 | 0.8233 |
| LinearSVC | SVM | 0.8233 | 0.8233 |
| MultinomialNB | Naive Bayes | 0.8050 | 0.8050 |

**UIT-VSFC — tiếng Việt, 3 lớp:**

| Mô hình | Họ | Accuracy | F1 (weighted) | F1 (macro) |
|---|---:|---:|---:|---:|
| **LinearSVC** | SVM | **0.8942** | **0.8866** | 0.6976 |
| SVC (RBF) | SVM | 0.8925 | 0.8804 | 0.6667 |
| MultinomialNB | Naive Bayes | 0.8758 | 0.8654 | 0.6639 |
| ComplementNB | Naive Bayes | 0.8742 | 0.8647 | 0.6594 |

> F1 (macro) thấp hơn F1 (weighted) vì lớp `neutral` ít mẫu hơn (mất cân bằng).

### 8.2 Các file kết quả sinh ra

**Thư mục `outputs/` (tiếng Anh) và `outputs_vn/` (tiếng Việt):**

| File | Ý nghĩa |
|---|---|
| `figures/01_eda.png` | Phân phối nhãn + độ dài câu (EDA) |
| `figures/02_model_comparison.png` | Bar chart so sánh 4 mô hình |
| `figures/03_confusion_matrices.png` | Ma trận nhầm lẫn 4 mô hình |
| `figures/04_top_features.png` | Từ quan trọng nhất theo từng lớp |
| `figures/05_speed_comparison.png` | So sánh thời gian train/predict |
| `figures/06_wordcloud.png` | Word cloud từ đặc trưng theo lớp |
| `tables/comparison_results.csv` | Bảng số liệu đầy đủ (dán vào báo cáo) |
| `tables/cross_validation_results.csv` | Kết quả cross-validation 5-fold |
| `tables/hyperparameter_tuning.csv` | Kết quả tuning (GridSearchCV) |

> Thư mục `outputs/` còn chứa vài file bổ sung từ notebook (ví dụ `00_real_data_eda.png`,
> `uci_model_comparison.csv`) — đó là kết quả phân tích sâu trên dữ liệu thật UCI.

## 9. Hướng dẫn thực nghiệm (tùy chỉnh)

### 9.1 Giới hạn số mẫu để chạy nhanh

Thêm `--sample` để chỉ dùng một phần dữ liệu (hữu ích khi thử nghiệm):

```powershell
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --source vn --data .\data\raw\vn_feedback.csv --sample 5000
```

### 9.2 Xuất kết quả sang thư mục khác (không ghi đè kết quả gốc)

Dùng `--output-dir` để giữ nguyên kết quả mẫu đã có:

```powershell
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --demo --output-dir outputs_thu_nghiem
```

### 9.3 Dùng dữ liệu riêng của bạn

Cách nhanh nhất: tạo file CSV theo định dạng mà bộ nạp `uci` mong đợi (3 cột
`text`, `sentiment`, `source`; nhãn `positive` / `negative`):

```csv
text,sentiment,source
"Sản phẩm rất tốt, giao nhanh",positive,my_data
"Chất lượng kém, không hài lòng",negative,my_data
"Dùng tạm được",positive,my_data
```

Rồi chạy:

```powershell
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --source uci --data .\data\raw\my_data.csv
```

> Bộ nạp `vn` **tự phát hiện** tên cột (`text`/`sentence`/`review`/`content` và
> `label`/`sentiment`/`class`) và tự chuyển nhãn `0/1/2` → `negative/neutral/positive`,
> nên bạn có thể dùng CSV 2 cột đơn giản với `--source vn`.

### 9.4 Chỉnh tham số mô hình

Mở `sentiment_svm_nb.py`, sửa hàm `build_pipelines()` (dòng ~264). Các tham số chính:

| Thành phần | Tham số hiện tại | Gợi ý thử |
|---|---|---|
| TF-IDF | `ngram_range=(1, 2)`, `max_features=50_000` | thử `(1, 3)` hoặc giảm `max_features` |
| MultinomialNB | `alpha=0.1` | thử `0.01`, `0.5`, `1.0` |
| ComplementNB | `alpha=1.0` | thử `0.5`, `2.0` |
| LinearSVC | `C=1.0` | thử `0.1`, `10.0` |
| SVC (RBF) | `C=1.0`, `gamma="scale"` | thử `gamma="auto"` |

Hàm `grid_search_tuning()` (dòng ~377) chứa lưới tham số `param_grids` — có thể mở rộng
để tìm tham số tối ưu tự động. Nếu muốn thử nhanh từng thay đổi, thao tác trực tiếp
trong **notebook** sẽ tiện hơn.

### 9.5 Đổi `random_state` / tỷ lệ train-test

Tỷ lệ split 80/20 và `random_state=42` nằm trong `main()` (dòng ~671) của
`sentiment_svm_nb.py`; trong notebook là cell `train_test_split`. Đổi giá trị này sẽ
tạo ra kết quả khác đôi chút (do dữ liệu chia lại khác).

---

## 10. Khắc phục lỗi thường gặp

| Lỗi | Nguyên nhân / cách xử lý |
|---|---|
| `python` không nhận diện | Cài Python 3.9–3.12 và **tick "Add to PATH"**. Hoặc dùng `py -3.11`. |
| `pip install` lỗi version | Nâng pip: `python -m pip install --upgrade pip`; nếu vẫn lỗi, xóa `.venv` tạo lại. |
| Không chạy được `Activate.ps1` | PowerShell chặn script: chạy `Set-ExecutionPolicy -Scope Process RemoteSigned` rồi active lại. |
| `FileNotFoundError` khi `--data` | Kiểm tra đúng đường dẫn file (chạy từ **thư mục gốc** dự án). |
| Lỗi tiếng Việt / `underthesea` | Cài `pip install underthesea`; chỉ ảnh hưởng `--source vn`. |
| Chữ tiếng Việt trên biểu đồ bị vuông (`□`) | Do thiếu font hệ thống; kết quả số không bị ảnh. Chỉ là hiển thị. |
| Kaggle báo lỗi 401 | Cấu hình API key: đặt biến `KAGGLE_USERNAME` / `KAGGLE_KEY`, hoặc tải file thủ công từ trang dataset. |
| Amazon chạy rất chậm / hết RAM | Dùng `--sample 50000` để giới hạn dữ liệu. |

---

## 11. Câu hỏi thường gặp (FAQ)

**Chạy mất bao lâu?**
- `--demo`: dưới 1 phút.
- UCI (3.000 câu): ~1–2 phút.
- Tiếng Việt (~16K câu): ~2–4 phút.
- Thời gian cụ thể tùy cấu hình máy.

**Kết quả có khớp với báo cáo không?**
Có, nếu chạy cùng dataset và `random_state=42`. Riêng cột *thời gian* train/predict là
phép đo cục bộ, dao động theo máy.

**Có cần GPU không?**
Không. Tất cả mô hình chạy trên CPU.

**Cần Python phiên bản bao nhiêu?**
3.9 đến 3.12. Khuyên dùng 3.10 / 3.11.

**Tôi có thể dùng dữ liệu riêng không?**
Có — xem mục 9.3 để biết định dạng CSV.

**Kết quả tốt nhất là mô hình nào?**
Trên UCI: `SVC (RBF)` (~0.83). Trên tiếng Việt: `LinearSVC` (~0.89). SVM thường nhỉnh
hơn Naive Bayes trên bài toán phân loại văn bản.

---

## 12. Trích dẫn & liên hệ

**Dataset & tài liệu tham khảo:**

- Kotzias, D., Denil, M., de Freitas, N., & Smyth, P. (2015). *From Group to Individual
  Labels Using Deep Features*. KDD 2015. UCI ML Repository, DOI `10.24432/C57604`.
- Nguyen et al. (2018). *UIT-VSFC: Vietnamese Students' Feedback Corpus for Sentiment
  Analysis*. KSE 2018, DOI `10.1109/KSE.2018.8573337`.
- Joachims, T. (1998). *Text Categorization with Support Vector Machines*.
- Maas, A. L., et al. (2011). *Learning Word Vectors for Sentiment Analysis*.

**Liên hệ:** xem thông tin tác giả trong `LICENSE`.

*Chúc bạn thao tác thuận lợi! Nếu có thắc mắc, hãy xem lại `README.md` (giới thiệu)
và `docs/theory.md` (cơ sở lý thuyết).*
