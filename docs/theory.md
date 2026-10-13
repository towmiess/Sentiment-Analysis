# Lý thuyết: SVM & Naive Bayes cho Sentiment Analysis
## Tài liệu tham khảo — đọc khi viết phần Cơ sở lý thuyết báo cáo

---

## 1. Bài toán Sentiment Analysis

**Định nghĩa:** Sentiment Analysis (Phân tích cảm xúc) là bài toán NLP phân loại
văn bản theo thái độ/cảm xúc của người viết.

**Phân loại:**
- Binary: Positive / Negative
- Multi-class: Positive / Negative / Neutral (phổ biến trong thực tế)
- Fine-grained: 1-5 sao (Very Neg / Neg / Neutral / Pos / Very Pos)

**Ứng dụng thực tế:**
- E-commerce: phân tích review sản phẩm (Amazon, Shopee, Tiki)
- Mạng xã hội: theo dõi brand reputation (Twitter, Facebook)
- Fintech: phân tích tin tức tài chính ảnh hưởng đến giá cổ phiếu
- Healthtech: phân tích phản hồi bệnh nhân về dịch vụ y tế

---

## 2. Naive Bayes

### 2.1 Định nghĩa và ý tưởng

**Naive Bayes** là họ các bộ phân loại **xác suất** (probabilistic classifier) dựa trên
**Định lý Bayes**, với giả định các đặc trưng (từ) **độc lập có điều kiện** với nhau.
Đây là một trong những thuật toán "cổ điển" nhất (Thomas Bayes, thế kỷ 18) nhưng vẫn là
baseline rất mạnh cho phân loại văn bản.

**Ý tưởng:** với mỗi văn bản, tính xác suất nó thuộc từng lớp sentiment rồi chọn lớp
có xác suất lớn nhất (quy tắc MAP — Maximum A Posteriori):

```
ŷ = argmax_C P(C | w1, w2, ..., wn)
```

### 2.2 Định lý Bayes

```
P(C | Text) = P(Text | C) × P(C) / P(Text)

Trong đó:
  P(C | Text)   : Posterior — xác suất Text thuộc lớp C khi đã quan sát các từ
  P(Text | C)   : Likelihood — xác suất sinh ra Text nếu biết trước lớp C
  P(C)          : Prior — tần suất lớp C trong tập huấn luyện
  P(Text)       : Evidence — hằng số chuẩn hóa, GIỐNG NHAU cho mọi C → bỏ qua
```

Vì `P(Text)` giống nhau cho mọi lớp khi so sánh, ta chỉ cần so sánh tử số:

```
P(C | Text) ∝ P(C) × P(Text | C)
```

### 2.3 Giả định "Naive" (độc lập có điều kiện)

Không gian văn bản là vô cùng lớn, không thể ước lượng `P(Text|C)` trực tiếp. NB giả
định các từ **độc lập với nhau khi đã biết lớp**:

```
P(Text | C) = P(w1|C) × P(w2|C) × ... × P(wn|C) = ∏ᵢ P(wi | C)
```

→ Luật phân loại trở thành:

```
ŷ = argmax_C [ P(C) × ∏ᵢ P(wi | C) ]
```

Giả định này **không đúng** trong thực tế ("not" đảo nghĩa "good", "good" và "movie"
thường đi cùng nhau), nhưng NB vẫn hiệu quả vì:
- Chỉ cần ước lượng từng `P(wi|C)` riêng lẻ → giảm mạnh số tham số, tránh "lời nguyền
  số chiều" (curse of dimensionality).
- Dù xác suất ước lượng bị lệch do bỏ qua tương quan, **thứ tự xếp hạng** giữa các lớp
  vẫn thường đúng → phân loại vẫn chính xác.

### 2.4 Luật phân loại & dạng log (tránh tràn số)

Nhân nhiều xác suất nhỏ (< 1) khiến tích tiến về 0 → tràn số (underflow). Thực tế dùng **log**:

```
log P(C | Text) ∝ log P(C) + Σᵢ log P(wi | C)
```

- Phép nhân → phép cộng → ổn định số học.
- `log` là hàm đơn điệu tăng nên chọn lớp có `log P` lớn nhất ⇔ chọn lớp có `P` lớn nhất.

### 2.5 Các biến thể Naive Bayes

| Biến thể | Phân phối | Dữ liệu phù hợp | Công thức P(wi\|C) |
|---|---|---|---|
| **MultinomialNB** | Multinomial | Text (count / TF-IDF) | `(N_ci + α) / (N_c + α·V)` |
| **ComplementNB** | Complement | Text mất cân bằng | Tham số từ phần bù các lớp khác |
| **BernoulliNB** | Bernoulli (0/1) | Binary features (từ có/không) | `p^xi · (1-p)^(1-xi)` |
| **GaussianNB** | Gaussian | Đặc trưng số liên tục | `N(μ_C, σ_C²)` |
| **CategoricalNB** | Categorical | Đặc trưng rời rạc | Tần suất từng giá trị |

**MultinomialNB** — phù hợp nhất với đếm từ (bag-of-words). Ước lượng xác suất có điều kiện:

```
P(wi | C) = (N_ci + α) / (N_c + α·V)
  N_ci : số lần từ wi xuất hiện trong lớp C
  N_c  : tổng số từ thuộc lớp C
  V    : kích thước từ điển (số từ duy nhất)
```

**ComplementNB** — cải tiến cho dữ liệu **mất cân bằng** (skewed). Thay vì ước lượng tham
số cho lớp C, nó ước lượng từ **phần bù** (tất cả các lớp KHÁC C):

```
θ'_ci = (Σ_{c'≠C} N_{c'i} + α) / (Σ_{c'≠C} N_{c'} + α·V)
```

rồi dùng `w_ci = log θ'_ci` và chuẩn hóa theo trọng số. Cách này giúp các từ hiếm nhưng
quan trọng không bị "đè bẹp" bởi lớp đông mẫu → thường vượt MultinomialNB trên dữ liệu lệch.

**BernoulliNB** — mỗi đặc trưng nhị phân (từ **có/không** xuất hiện, bỏ qua tần suất).
Hợp khi sự *xuất hiện* của từ quan trọng hơn số lần lặp (ví dụ phát hiện spam).

**GaussianNB** — giả định đặc trưng liên tục theo phân phối chuẩn. Ít dùng cho text (TF-IDF
là sparse) nhưng phổ biến cho dữ liệu số/tabular.

### 2.6 Laplace / Lidstone Smoothing

Nếu từ `wi` xuất hiện ở lớp khác nhưng **không** xuất hiện ở lớp C → `P(wi|C) = 0` →
`∏ P(wi|C) = 0` (mô hình gán xác suất 0 cho cả văn bản). Smoothing cộng `α` vào tử và
`α·V` vào mẫu để "mượn" xác suất cho từ chưa thấy:

```
P(wi | C) = (count(wi, C) + α) / (Σ count(w, C) + α·|V|)
  α = 1.0 : Laplace smoothing (chuẩn)
  α = 0.1 : Lidstone smoothing (nhẹ hơn, thường hợp TF-IDF)
```

### 2.7 Ưu / Nhược điểm

**Ưu điểm:**
- Huấn luyện `O(n·d)` — rất nhanh, không cần vòng lặp tối ưu (closed-form).
- Học tốt với ít dữ liệu, ít overfit.
- Hiệu quả với text ngắn (SMS, tweet, review).
- Cho ra **xác suất** → dễ hiểu, dễ kết hợp ensemble.
- Hỗ trợ **học online/incremental** (`partial_fit`) — cập nhật khi có dữ liệu mới.

**Nhược điểm:**
- Giả định độc lập sai → kém với quan hệ từ và phủ định ("not good").
- Nhạy với feature có xác suất 0 (cần smoothing).
- Xác suất đầu ra thường **overconfident** (bị lệch) → tốt cho ranking hơn calibrated probability.

### 2.8 Ứng dụng thực tế của Naive Bayes

- **Lọc thư rác (spam filtering)**: bài toán kinh điển (Paul Graham "A Plan for Spam"),
  dùng MultinomialNB/BernoulliNB trên nội dung email.
- **Phân loại chủ đề văn bản**: Reuters, 20 Newsgroups, gán tag tin tức.
- **Sentiment analysis**: phân loại review/tweet positive vs negative.
- **Phát hiện ngôn ngữ, phân loại email tự động, recommendation giai đoạn đầu** (cold start).

---

## 3. Support Vector Machine (SVM)

### 3.1 Định nghĩa và ý tưởng

**Support Vector Machine (SVM)** là thuật toán phân loại **học có giám sát**, do Vladimir
Vapnik & Corinna Cortes giới thiệu (1995), tìm **siêu phẳng tối ưu** phân tách các lớp với
**margin (lề) lớn nhất**. Khác với perceptron/logistic regression (chỉ cần phân đúng), SVM
tìm mặt phẳng **cách xa cả hai lớp nhất** → tổng quát hóa tốt hơn trên dữ liệu mới.

```
Siêu phẳng:   w·x + b = 0
  w : vector pháp tuyến (trọng số)
  b : hệ số bias

Quy tắc dự đoán:   ŷ = sign(w·x + b)
Margin (lề):       2 / ||w||
```

**Ý tưởng:** margin lớn nhất ⇔ `||w||` nhỏ nhất, đồng thời mọi điểm được phân đúng lớp.

### 3.2 Bài toán tối ưu — công thức đầy đủ

**Dạng gốc (Primal)** với dữ liệu tách được tuyến tính (hard margin):

```
Minimize    (1/2) ||w||²
subject to  yi (w·xi + b) ≥ 1,  ∀i
```

**Dạng đối ngẫu (Dual)** — giải hiệu quả hơn và cho phép dùng kernel. Dùng nhân tử
Lagrange `αᵢ ≥ 0`:

```
Maximize    Σᵢ αᵢ - (1/2) Σᵢ Σⱼ αᵢ αⱼ yi yj (xi·xj)
subject to  Σᵢ αᵢ yi = 0,   αᵢ ≥ 0
```

Điểm mấu chốt: hàm mục tiêu đối ngẫu **chỉ phụ thuộc tích vô hướng** `xi·xj` → đây chính
là nơi kernel trick thay `xi·xj` bằng `K(xi, xj)`. Nghiệm: `w = Σᵢ αᵢ yi xi` — chỉ các điểm
có `αᵢ > 0` (support vectors) đóng góp vào `w`.

### 3.3 Support Vectors

Các điểm nằm đúng trên lề (`yi(w·xi + b) = 1`, tức `αᵢ > 0`) gọi là **Support Vectors**.
Chúng là toàn bộ "bộ nhớ" của mô hình: bỏ mọi điểm khác, siêu phẳng không thay đổi. Với
text, support vectors thường là các câu "khó" nằm gần biên (có phủ định, cảm xúc trái chiều).

### 3.4 Soft Margin — biến slack ξ và tham số C

Dữ liệu thực tế hiếm khi tách được tuyến tính hoàn toàn. Soft-margin cho phép một số điểm
vi phạm lề bằng biến slack `ξᵢ ≥ 0`:

```
Minimize    (1/2) ||w||² + C · Σᵢ ξᵢ
subject to  yi (w·xi + b) ≥ 1 - ξᵢ,   ξᵢ ≥ 0
```

- `C` cân bằng giữa **margin rộng** (hạng tử đầu) và **số lỗi** (hạng tử sau).
- `C` lớn → phạt lỗi nặng → fit chặt, ít lỗi train nhưng dễ **overfit**.
- `C` nhỏ → chấp nhận lỗi → margin rộng → **generalize** tốt hơn.
- `C = 1.0` là mặc định, thường tốt với text sau TF-IDF.

### 3.5 Kernel Trick

Với dữ liệu phi tuyến, thay vì biến đổi tường minh `φ(x)` (có thể vô hạn chiều), ta chỉ cần
tính **hàm kernel** `K(xi, xj) = φ(xi)·φ(xj)` — tương đương tích vô hướng trong không gian
đặc trưng mà KHÔNG cần xây `φ(x)`. Điều kiện: kernel thỏa **định lý Mercer** (ma trận kernel
bán xác định dương).

| Kernel | Công thức | Dùng khi |
|---|---|---|
| **Linear** | `xi · xj` | Text (TF-IDF gần như tuyến tính) |
| **RBF (Gaussian)** | `exp(-γ·||xi - xj||²)` | Dữ liệu phi tuyến, tabular |
| **Polynomial** | `(γ·xi·xj + r)^d` | Ít dùng với text |
| **Sigmoid** | `tanh(γ·xi·xj + r)` | Ít dùng (mô phỏng neural 2 lớp) |

**Với text (TF-IDF): kernel tuyến tính (LinearSVC) gần như luôn tốt nhất** vì không gian
TF-IDF nhiều chiều đã gần như tách được tuyến tính:
- Linear SVM: `O(n·d)` — nhanh, scale tốt.
- RBF kernel: `O(n²·d)` — rất chậm với dữ liệu lớn, dễ overfit nếu `γ` lớn.

### 3.6 Các siêu tham số quan trọng

| Tham số | Ý nghĩa | Ảnh hưởng |
|---|---|---|
| `C` | Hệ số phạt (regularization ngược) | Lớn → overfit; nhỏ → underfit |
| `gamma` (γ) | Độ rộng RBF kernel | Lớn → boundary phức tạp, overfit; nhỏ → underfit |
| `degree` | Bậc polynomial kernel | Cao → boundary phức tạp |
| `coef0` | Hệ số `r` của poly/sigmoid | Ảnh hưởng độ cong |
| `class_weight` | Trọng số theo lớp | Xử lý dữ liệu mất cân bằng |

Với RBF: `gamma='scale'` (mặc định sklearn) = `1 / (n_features · X.var())` — tự co giãn theo
thang dữ liệu, thay cho `gamma='auto'` = `1 / n_features` cũ.

### 3.7 Các biến thể SVM trong scikit-learn

| Lớp | Solver | Đặc điểm |
|---|---|---|
| **LinearSVC** | liblinear | Tuyến tính, rất nhanh, tối ưu cho text/sparse; mặc định OvR |
| **SVC** | libsvm | Hỗ trợ mọi kernel; `probability=True` dùng Platt scaling |
| **NuSVC** | libsvm | Điều khiển qua tham số `ν` thay cho `C` (giới hạn tỉ lệ support vectors/lỗi) |
| **SVR** | libsvm | Hồi quy — SVM cho mục tiêu liên tục |

```python
from sklearn.svm import LinearSVC, SVC, NuSVC, SVR

LinearSVC(C=1.0, max_iter=2000)          # liblinear — NHANH, dùng cho text
SVC(kernel='rbf', C=1.0, gamma='scale')  # libsvm — mọi kernel
NuSVC(nu=0.5)                            # điều khiển qua nu thay vì C
SVR(kernel='rbf')                        # hồi quy
```

Lưu ý: `LinearSVC` (liblinear) nhanh hơn nhiều so với `SVC(kernel='linear')` (libsvm) trên
text lớn, vì liblinear tối ưu riêng cho bài toán tuyến tính.

### 3.8 Multi-class SVM

SVM gốc là **binary** classifier. Hai chiến lược mở rộng đa lớp:

- **OvR (One-vs-Rest / One-vs-All):** huấn luyện `k` mô hình, mỗi mô hình phân biệt 1 lớp
  với tất cả các lớp còn lại. `LinearSVC` dùng OvR mặc định.
- **OvO (One-vs-One):** huấn luyện `k(k-1)/2` mô hình, mỗi mô hình phân biệt 2 lớp; lớp
  thắng nhiều phiếu nhất (voting) là kết quả. `SVC` dùng OvO mặc định.

OvO có nhiều mô hình hơn nhưng mỗi mô hình nhỏ; với text (nhiều lớp + sparse) OvR thường
được ưa chuộng vì ít mô hình và tuyến tính đã đủ mạnh.

### 3.9 Ưu / Nhược điểm

**Ưu điểm:**
- Rất mạnh với dữ liệu **nhiều chiều, thưa (sparse)** như TF-IDF.
- **Ít overfit** nhờ tối đa hóa margin (regularization cấu trúc).
- Kernel trick xử lý phi tuyến linh hoạt.
- **Giải thích được** qua `coef_` (kernel tuyến tính) — xem top từ cho mỗi lớp.
- Không cần chuẩn hóa đặc trưng khi dùng TF-IDF (vector thưa đã chuẩn hóa L2).

**Nhược điểm:**
- Chậm hơn NB, đặc biệt RBF (`O(n²·d)`), khó scale với hàng triệu mẫu.
- Không cho **xác suất** trực tiếp (cần `CalibratedClassifierCV` / Platt scaling).
- Nhạy với `C`, `γ` → cần tuning (GridSearch).
- Kém minh bạch với kernel phi tuyến (không đọc trực tiếp trọng số từ).

### 3.10 Ứng dụng thực tế của SVM

- **Text classification / sentiment analysis**: LinearSVC là lựa chọn hàng đầu trên TF-IDF.
- **Nhận dạng ảnh & chữ viết tay** (trước thời deep learning): MNIST, nhận dạng khuôn mặt.
- **Bioinformatics**: phân loại gene/protein, ung thư (microarray) — nơi số chiều >> số mẫu.
- **Tài chính**: phát hiện gian lận, dự báo xu hướng giá từ đặc trưng số.

---

## 4. TF-IDF — Biểu diễn văn bản thành vector số

### 4.1 Vì sao cần TF-IDF?

Máy học không hiểu văn bản thô → cần chuyển text thành vector số. Hai cách phổ biến:

- **Bag-of-Words (`CountVectorizer`)**: đếm tần suất từ.
- **TF-IDF**: trọng số hóa tần suất để giảm ảnh hưởng của từ quá phổ biến.

### 4.2 Công thức

```
TF(t, d)      = số lần từ t xuất hiện trong tài liệu d
IDF(t)        = log( (1 + N) / (1 + df(t)) ) + 1
TF-IDF(t, d)  = TF(t, d) × IDF(t, d)
```

Trong đó:

- `N`: tổng số tài liệu.
- `df(t)`: số tài liệu chứa từ `t`.

Từ càng hiếm (`df` nhỏ) → IDF càng lớn → trọng số cao hơn. Từ phổ biến (như "the", "is") → trọng số thấp. Tham số `sublinear_tf=True` dùng `1 + log(TF)` thay vì `TF` để giảm ảnh hưởng của text dài.

### 4.3 Tại sao dùng bigram (`ngram_range=(1,2)`)?

- **Unigram** chỉ bắt từng từ đơn → bỏ lỡ ngữ nghĩa cụm như `not good`, `not bad`.
- **Bigram** giữ lại cặp 2 từ liên tiếp → nhận diện phủ định: `not_good` trở thành một feature riêng.
- `ngram_range=(1,2)` kết hợp cả unigram + bigram → cân bằng giữa phủ từ vựng và ngữ nghĩa cụm.

---

## 5. Metrics đánh giá mô hình phân loại

### 5.1 Các chỉ số cơ bản

| Ký hiệu | Tên | Công thức | Ý nghĩa |
|---|---|---|---|
| Acc | Accuracy | (TP+TN) / Tổng | Tỉ lệ dự đoán đúng |
| P | Precision | TP / (TP+FP) | Trong các dự đoán Positive, bao nhiêu đúng |
| R | Recall | TP / (TP+FN) | Trong các Positive thật, bắt được bao nhiêu |
| F1 | F1-Score | 2·P·R / (P+R) | Trung bình điều hòa của P và R |

### 5.2 Confusion Matrix

Ma trận nhầm lẫn liệt kê 4 trường hợp:

- **TP** (True Positive): Positive thật, đoán đúng Positive.
- **TN** (True Negative): Negative thật, đoán đúng Negative.
- **FP** (False Positive): Negative thật, đoán nhầm Positive (sai loại 1).
- **FN** (False Negative): Positive thật, đoán nhầm Negative (sai loại 2).

### 5.3 Tại sao F1 quan trọng hơn Accuracy khi dữ liệu lệch?

Với dữ liệu imbalanced (ví dụ 95% negative), mô hình đoán mọi thứ là negative vẫn đạt Accuracy 95% nhưng Recall của positive = 0 → vô dụng. F1 (trung bình điều hòa P và R) phạt nặng khi P hoặc R thấp → phản ánh đúng chất lượng.

- **Macro-F1**: trung bình F1 của từng lớp → công bằng giữa các lớp.
- **Weighted-F1**: trung bình F1 có trọng số theo số mẫu (support).

---

## 6. So sánh trực tiếp với Sentiment Task

| Tình huống | NB | SVM |
|---|---|---|
| "This is good" | ✅ Pos | ✅ Pos |
| "This is not good" | ❌ có thể sai (not × good) | ✅ Pos (bigram "not good") |
| "not bad at all" | ❌ dễ nhầm | ✅ thường đúng |
| "I expected more" | ❌ khó | ❌ khó (cả hai) |
| Text rất ngắn (< 5 từ) | ✅ vẫn OK | ⚠️ thiếu features |
| Text rất dài (> 500 từ) | ✅ vẫn nhanh | ✅ LinearSVC OK |
| Dataset 1K mẫu | ✅ tốt | ⚠️ cần đủ data |
| Dataset 50K+ mẫu | ✅ | ✅ SVM tốt hơn rõ rệt |

---

## 7. Feature Importance — Giải thích mô hình

### Naive Bayes — feature_log_prob_
```python
# Top từ mỗi lớp theo log probability
feature_names = vectorizer.get_feature_names_out()
for i, class_name in enumerate(classes):
    top_idx = np.argsort(nb_model.feature_log_prob_[i])[-15:][::-1]
    print(f"{class_name}: {', '.join(feature_names[top_idx])}")
```

### LinearSVC — coef_
```python
# coef_[i] = trọng số của mỗi feature với lớp i
# coef_ dương → feature push về lớp này
# coef_ âm → feature push về lớp khác
feature_names = vectorizer.get_feature_names_out()
for i, class_name in enumerate(classes):
    top_pos = np.argsort(svm_model.coef_[i])[-15:][::-1]  # Top Positive
    top_neg = np.argsort(svm_model.coef_[i])[:15]          # Top Negative
    print(f"[{class_name}] Pos features: {', '.join(feature_names[top_pos])}")
    print(f"[{class_name}] Neg features: {', '.join(feature_names[top_neg])}")
```

---

## 8. KẾT QUẢ THỰC NGHIỆM

### 8.1 Dữ liệu và cấu hình

- Dataset: UCI Sentiment Labelled Sentences (3.000 review, cân bằng 1.500 pos / 1.500 neg).
- Chia 80/20 stratified (`random_state=42`): 2.400 train / 600 test.
- TF-IDF: `ngram_range=(1,2)`, `max_features=50.000`, `sublinear_tf=True`, `min_df=2`.
- 4 mô hình: MultinomialNB, ComplementNB, LinearSVC, SVC(RBF).

### 8.2 Bảng so sánh hiệu năng

| Mô hình | Họ | Accuracy | F1 (weighted) | F1 (macro) | Train (s) | Predict (ms) |
|---|---:|---:|---:|---:|---:|---:|
| SVC (RBF) | SVM | 0.8283 | 0.8283 | 0.8283 | 0.62 | 121.5 |
| ComplementNB | Naive Bayes | 0.8233 | 0.8233 | 0.8233 | 0.07 | 9.5 |
| LinearSVC | SVM | 0.8233 | 0.8233 | 0.8233 | 0.09 | 10.0 |
| MultinomialNB | Naive Bayes | 0.8050 | 0.8050 | 0.8050 | 0.12 | 13.6 |

### 8.3 Ma trận nhầm lẫn (TN, FP, FN, TP)

| Mô hình | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| SVC (RBF) | 255 | 45 | 58 | 242 |
| ComplementNB | 249 | 51 | 55 | 245 |
| LinearSVC | 247 | 53 | 53 | 247 |
| MultinomialNB | 246 | 54 | 63 | 237 |

### 8.4 Chỉ số theo lớp (classification report)

| Mô hình | Lớp | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| SVC (RBF) | negative | 0.8147 | 0.8500 | 0.8320 |
| SVC (RBF) | positive | 0.8432 | 0.8067 | 0.8245 |
| LinearSVC | negative | 0.8233 | 0.8233 | 0.8233 |
| LinearSVC | positive | 0.8233 | 0.8233 | 0.8233 |
| ComplementNB | negative | 0.8191 | 0.8300 | 0.8245 |
| ComplementNB | positive | 0.8277 | 0.8167 | 0.8221 |
| MultinomialNB | negative | 0.7961 | 0.8200 | 0.8079 |
| MultinomialNB | positive | 0.8144 | 0.7900 | 0.8020 |

### 8.5 Demo inference

- `"not bad at all, actually quite decent"` → **positive** ✅
- `"Oh great, another bug..."` → **positive** (sai do sarcasm)
- `"The product arrived on time. Nothing special."` → **negative** (nhãn thật trung tính)

### 8.6 Cross-validation (Stratified 5-Fold)

Để tránh phụ thuộc một split cố định, chạy **Stratified K-Fold** (k = 5). Kết quả `mean ± std`:

| Mô hình | Accuracy | F1 (weighted) | F1 (macro) |
|---|---:|---:|---:|
| MultinomialNB | 0.8250 ± 0.0091 | 0.8250 ± 0.0091 | 0.8250 ± 0.0091 |
| ComplementNB | 0.8293 ± 0.0065 | 0.8293 ± 0.0064 | 0.8293 ± 0.0064 |
| LinearSVC | 0.8337 ± 0.0046 | 0.8336 ± 0.0046 | 0.8336 ± 0.0046 |
| SVC (RBF) | 0.8383 ± 0.0072 | 0.8381 ± 0.0072 | 0.8381 ± 0.0072 |

Thứ hạng giữ nguyên (SVC RBF > LinearSVC > ComplementNB > MultinomialNB) với độ lệch
chuẩn nhỏ (≤ 0.01) → kết quả ổn định, không phải may rủi của một lần chia.

### 8.7 Hyperparameter tuning (GridSearchCV)

GridSearchCV (3-fold, scoring = F1-weighted):

| Mô hình | Best params | CV F1 (weighted) |
|---|---:|---:|
| MultinomialNB | alpha=0.5 | 0.8153 |
| ComplementNB | alpha=2.0 | 0.8199 |
| LinearSVC | C=0.1 | 0.8236 |
| SVC (RBF) | C=1.0, gamma=scale | 0.8199 |

Các giá trị tối ưu gần với mặc định nên bảng so sánh chính (§8.2) gần như không đổi.

---

## 9. Kết quả tiếng Việt (UIT-VSFC)

### 9.1 Dữ liệu và cấu hình

- Dataset: UIT Vietnamese Students' Feedback (16.175 câu, 3 lớp `negative / neutral / positive`).
- Dùng 6.000 mẫu ngẫu nhiên (`random_state=42`) để giữ thời gian chạy hợp lý.
- Tiền xử lý: `underthesea.word_tokenize` + bỏ URL/ký tự đặc biệt.
- TF-IDF và 4 mô hình như §8.

### 9.2 Bảng so sánh hiệu năng

| Mô hình | Họ | Accuracy | F1 (weighted) | F1 (macro) | Train (s) | Predict (ms) |
|---|---:|---:|---:|---:|---:|---:|
| LinearSVC | SVM | 0.8942 | 0.8866 | 0.6976 | 0.29 | 40.1 |
| SVC (RBF) | SVM | 0.8925 | 0.8804 | 0.6667 | 6.0 | 1031.7 |
| MultinomialNB | Naive Bayes | 0.8758 | 0.8654 | 0.6639 | 0.33 | 57.8 |
| ComplementNB | Naive Bayes | 0.8742 | 0.8647 | 0.6594 | 0.31 | 54.2 |

### 9.3 Nhận xét

- **LinearSVC** dẫn đầu (F1-weighted = 0.8866), vượt cả SVC (RBF) — kernel tuyến tính đủ
  mạnh và nhanh hơn hẳn (Predict 40 ms vs 1032 ms).
- **F1 macro** thấp hơn nhiều (≈ 0.66–0.70) do lớp **neutral** chỉ chiếm ~4% mẫu.
- Naive Bayes thấp hơn SVM ~2 điểm F1-weighted — cùng xu hướng với tiếng Anh.
- So với UCI 2 lớp, bài toán 3 lớp tiếng Việt khó hơn (F1 macro giảm mạnh) nhưng
  F1 weighted vẫn cao do hai lớp positive/negative chiếm đa số.

---

## 10. KẾT LUẬN CHI TIẾT

### 10.1 Mô hình tốt nhất

- **SVC (RBF)** dẫn đầu (F1 = 0.8283), cao hơn nhóm còn lại ~0.5 điểm %.
- **LinearSVC** và **ComplementNB** đồng hạng (0.8233) — LinearSVC cân bằng tuyệt đối giữa 2 lớp (P = R = F1 = 0.8233), ComplementNB dự đoán nhanh nhất (8.5 ms).
- **MultinomialNB** kém nhất (0.8050), đúng như dự đoán lý thuyết vì không xử lý tốt phủ định.

### 10.2 SVM vs Naive Bayes — rút ra từ kết quả

1. **Về độ chính xác**: SVM thắng NB nhờ tối ưu margin trong không gian TF-IDF; bigram giúp bắt cụm phủ định. SVC(RBF) cho F1 cao nhất.
2. **Về tốc độ**: NB thắng tuyệt đối — huấn luyện ~0.07s, dự đoán ~9.5ms (nhanh hơn SVC(RBF) ~13 lần về dự đoán).
3. **Về cân bằng chi phí**: nếu cần vừa nhanh vừa tốt, **LinearSVC** (0.8233, 10ms) hoặc **ComplementNB** (0.8233, 9.5ms) là lựa chọn thực dụng.

### 10.3 Hạn chế

- Bag-of-Words bỏ qua thứ tự từ và ngữ cảnh → chưa xử lý sarcasm, emoji, neutral.
- Kết quả trên một dataset nhỏ và cân bằng; cross-validation (mục 8.6) cho độ lệch chuẩn nhỏ nhưng kết luận vẫn giới hạn trong miền review ngắn.
- RBF kernel chậm hơn nhiều với dữ liệu lớn.

### 10.4 Hướng phát triển

- Dùng word embeddings (Word2Vec/GloVe) hoặc fine-tune BERT để giữ ngữ nghĩa.
- Xử lý mất cân bằng lớp (neutral) bằng class_weight hoặc oversampling cho bài toán 3 lớp.
- Fine-tune PhoBERT cho tiếng Việt và đánh giá trên toàn bộ 16K câu.
