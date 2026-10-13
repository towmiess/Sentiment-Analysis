"""
=============================================================
Sentiment Analysis: SVM vs Naive Bayes — Pipeline Hoàn Chỉnh
=============================================================
Hỗ trợ 3 dataset:
  --source imdb    : IMDB 50K Movie Reviews (2 lớp)
  --source amazon  : Amazon Reviews (3 lớp — khuyên dùng cho báo cáo)
  --source vn      : Vietnamese Student Feedback (3 lớp, Tiếng Việt)

Cách chạy:
  python sentiment_svm_nb.py --source imdb   --data imdb_dataset.csv
  python sentiment_svm_nb.py --source amazon --data train.ft.txt
  python sentiment_svm_nb.py --source vn     --data vn_feedback.csv
  python sentiment_svm_nb.py --demo          # 5000 mẫu IMDB, test nhanh

Kaggle download:
  kaggle datasets download -d lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
  kaggle datasets download -d bittlingmayer/amazonreviews

HuggingFace (VN):
  from datasets import load_dataset
  ds = load_dataset("uitnlp/vietnamese_students_feedback")
  ds["train"].to_csv("vn_feedback.csv", index=False)
=============================================================
"""

import argparse, os, sys, time, re, warnings
from pathlib import Path
warnings.filterwarnings("ignore")

os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.getcwd(), ".matplotlib"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

FIGURES_DIR = Path("outputs") / "figures"
TABLES_DIR = Path("outputs") / "tables"


def configure_output_dirs(output_dir: str | Path = "outputs") -> tuple[Path, Path]:
    """Create and configure folders used for figures and result tables."""
    global FIGURES_DIR, TABLES_DIR
    output_path = Path(output_dir)
    FIGURES_DIR = output_path / "figures"
    TABLES_DIR = output_path / "tables"
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    return FIGURES_DIR, TABLES_DIR

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from nltk.corpus import stopwords
from nltk.stem  import PorterStemmer

from sklearn.model_selection     import (
    train_test_split, StratifiedKFold, cross_val_score, GridSearchCV,
)
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes         import MultinomialNB, ComplementNB
from sklearn.svm                 import LinearSVC, SVC
from sklearn.pipeline            import Pipeline
from sklearn.preprocessing       import LabelEncoder
from sklearn.metrics             import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix,
)

# ════════════════════════════════════════════════════════════
# 1. LOAD DỮ LIỆU
# ════════════════════════════════════════════════════════════

def load_imdb(path: str, sample: int = None) -> tuple:
    """
    IMDB: CSV với cột 'review' và 'sentiment' (positive/negative)
    """
    df = pd.read_csv(path)
    if sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)

    label_names = sorted(df["sentiment"].unique().tolist())
    le = LabelEncoder().fit(label_names)
    y  = le.transform(df["sentiment"])

    print(f"✅ IMDB loaded: {len(df):,} rows | Labels: {label_names}")
    print(f"   Distribution:\n{df['sentiment'].value_counts().to_string()}\n")
    return df["review"].tolist(), y, label_names


def load_uci(path: str, sample: int = None) -> tuple:
    """Load UCI Sentiment Labelled Sentences CSV: text, sentiment, source."""
    df = pd.read_csv(path)
    required_columns = {"text", "sentiment", "source"}
    missing_columns = required_columns - set(df.columns)
    if missing_columns:
        raise ValueError(
            "UCI CSV must contain columns: text, sentiment, source. "
            f"Missing: {sorted(missing_columns)}"
        )

    df = df.dropna(subset=["text", "sentiment"]).copy()
    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)

    label_names = sorted(df["sentiment"].unique().tolist())
    le = LabelEncoder().fit(label_names)
    y = le.transform(df["sentiment"])

    print(f"UCI loaded: {len(df):,} rows | Labels: {label_names}")
    print(f"   Distribution:\n{df['sentiment'].value_counts().to_string()}\n")
    return df["text"].tolist(), y, label_names


def load_amazon(path: str, sample: int = None) -> tuple:
    """
    Amazon .ft.txt format: __label__1 hoặc __label__2 + text
    Label 1 = Neg (1-2 sao), Label 2 = Pos (4-5 sao)
    Gom thành 3 lớp bằng cách lấy thêm mẫu "neutral" từ 3 sao (nếu có)
    hoặc đơn giản dùng 2 lớp từ file gốc.
    """
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            label = "positive" if "__label__2" in line else "negative"
            text  = re.sub(r"__label__\d", "", line).strip()
            rows.append({"text": text, "sentiment": label})
    df = pd.DataFrame(rows)

    if sample:
        df = df.sample(n=sample, random_state=42).reset_index(drop=True)

    label_names = sorted(df["sentiment"].unique().tolist())
    le = LabelEncoder().fit(label_names)
    y  = le.transform(df["sentiment"])

    print(f"✅ Amazon loaded: {len(df):,} rows | Labels: {label_names}")
    print(f"   Distribution:\n{df['sentiment'].value_counts().to_string()}\n")
    return df["text"].tolist(), y, label_names


def load_vietnamese(path: str, sample: int = None) -> tuple:
    """
    UIT Vietnamese Students Feedback CSV
    Columns: sentence, sentiment (0=NEG, 1=NEU, 2=POS)
    hoặc: text, label
    """
    df = pd.read_csv(path)
    print(f"   Columns detected: {list(df.columns)}")

    # Tự động detect cột
    text_col  = next((c for c in df.columns
                      if any(k in c.lower() for k in ["text","sentence","review","content"])),
                     df.columns[0])
    label_col = next((c for c in df.columns
                      if any(k in c.lower() for k in ["label","sentiment","class"])),
                     df.columns[-1])

    # Map số sang tên nếu cần
    label_map = {0: "negative", 1: "neutral", 2: "positive",
                 "0": "negative", "1": "neutral", "2": "positive"}
    if df[label_col].dtype in [int, "int64"] or str(df[label_col].iloc[0]) in label_map:
        df[label_col] = df[label_col].map(label_map).fillna(df[label_col].astype(str))

    if sample:
        df = df.sample(n=min(sample, len(df)), random_state=42).reset_index(drop=True)

    label_names = sorted(df[label_col].unique().tolist())
    le = LabelEncoder().fit(label_names)
    y  = le.transform(df[label_col])

    print(f"✅ Vietnamese loaded: {len(df):,} rows | Labels: {label_names}")
    print(f"   Distribution:\n{df[label_col].value_counts().to_string()}\n")
    return df[text_col].tolist(), y, label_names


def load_demo() -> tuple:
    """Tạo IMDB sample nhỏ để test nhanh không cần file."""
    try:
        from sklearn.datasets import load_files
    except ImportError:
        pass
    # Fallback: dùng mẫu tự tạo
    texts = [
        "Absolutely wonderful film, touching and beautiful.",
        "Great acting, brilliant screenplay, must watch!",
        "One of the best movies I have ever seen.",
        "Amazing performances by the entire cast.",
        "Heartwarming story, highly recommended for everyone.",
        "Terrible movie, waste of time and money.",
        "Boring plot, bad acting, very disappointed.",
        "I hated every single minute of this film.",
        "Worst movie of the year, avoid it.",
        "Completely pointless, nothing interesting at all.",
    ] * 250   # 2500 pos + 2500 neg = 5000 mẫu
    y = ([1] * 5 + [0] * 5) * 250
    label_names = ["negative", "positive"]
    print(f"✅ Demo mode: {len(texts)} synthetic samples")
    return texts, np.array(y), label_names


# ════════════════════════════════════════════════════════════
# 2. TIỀN XỬ LÝ
# ════════════════════════════════════════════════════════════

_stemmer   = PorterStemmer()
try:
    _stopwords = set(stopwords.words("english")) - {"no", "nor", "not"}
except LookupError:
    _stopwords = {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "by",
        "for", "from", "has", "have", "he", "her", "hers", "him", "his", "i",
        "in", "is", "it", "its", "me", "my", "of", "on", "or", "our", "ours",
        "she", "so", "that", "the", "their", "theirs", "them", "they", "this",
        "to", "was", "we", "were", "what", "when", "where", "which", "who", "will",
        "with", "you", "your", "yours"
    }
    print("NLTK stopwords not found; using built-in English stopwords fallback.")


def clean_en(text: str) -> str:
    """Tiền xử lý Tiếng Anh."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>",  " ", text)   # HTML tags
    text = re.sub(r"http\S+",  " ", text)   # URLs
    text = re.sub(r"[^a-z\s]", " ", text)   # Chỉ giữ chữ cái
    text = re.sub(r"\s+",      " ", text).strip()
    tokens = [_stemmer.stem(w) for w in text.split()
              if w not in _stopwords and len(w) > 2]
    return " ".join(tokens)


def clean_vi(text: str) -> str:
    """Tiền xử lý Tiếng Việt (cần underthesea)."""
    try:
        from underthesea import word_tokenize
        text = str(text)
        text = re.sub(r"http\S+", " ", text)
        text = re.sub(r"[^\w\s]", " ", text)
        tokens = word_tokenize(text.lower(), format="text").split()
        return " ".join(t for t in tokens if len(t) > 1)
    except ImportError:
        print("⚠️  underthesea không có — dùng clean đơn giản cho Tiếng Việt")
        text = re.sub(r"[^\w\s]", " ", str(text).lower())
        return " ".join(text.split())


def preprocess(texts: list, lang: str = "en") -> list:
    clean_fn = clean_vi if lang == "vi" else clean_en
    print(f"🔄 Preprocessing {len(texts):,} texts (lang={lang})...")
    t0 = time.time()
    result = [clean_fn(t) for t in texts]
    print(f"   Done in {time.time()-t0:.1f}s\n")
    return result


# ════════════════════════════════════════════════════════════
# 3. XÂY DỰNG PIPELINES
# ════════════════════════════════════════════════════════════

def build_pipelines() -> dict:
    """
    4 mô hình để so sánh:
      2 Naive Bayes variants + 2 SVM variants
    """
    tfidf_params = dict(
        ngram_range=(1, 2),
        max_features=50_000,
        sublinear_tf=True,
        min_df=2,
    )
    return {
        "MultinomialNB": Pipeline([
            ("vec", TfidfVectorizer(**tfidf_params)),
            ("clf", MultinomialNB(alpha=0.1)),
        ]),
        "ComplementNB": Pipeline([
            ("vec", TfidfVectorizer(**tfidf_params)),
            ("clf", ComplementNB(alpha=1.0)),
        ]),
        "LinearSVC": Pipeline([
            ("vec", TfidfVectorizer(**tfidf_params)),
            ("clf", LinearSVC(C=1.0, max_iter=2000, random_state=42)),
        ]),
        "SVC (RBF)": Pipeline([
            ("vec", TfidfVectorizer(**tfidf_params)),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)),
        ]),
    }


# ════════════════════════════════════════════════════════════
# 4. TRAIN & ĐÁNH GIÁ
# ════════════════════════════════════════════════════════════

def train_compare(X_tr, X_te, y_tr, y_te,
                  pipelines: dict, label_names: list) -> tuple:
    records, trained = [], {}

    print("=" * 65)
    print("🚀 TRAINING & EVALUATING")
    print("=" * 65)

    for name, pipe in pipelines.items():
        print(f"\n▶ {name}")

        t0 = time.time()
        pipe.fit(X_tr, y_tr)
        t_train = time.time() - t0

        t1 = time.time()
        y_pred = pipe.predict(X_te)
        t_pred = (time.time() - t1) * 1000

        acc = accuracy_score(y_te, y_pred)
        f1w = f1_score(y_te, y_pred, average="weighted")
        f1m = f1_score(y_te, y_pred, average="macro")

        print(f"   Accuracy      : {acc:.4f} ({acc*100:.2f}%)")
        print(f"   F1 (weighted) : {f1w:.4f}")
        print(f"   F1 (macro)    : {f1m:.4f}")
        print(f"   Train: {t_train:.2f}s  |  Predict: {t_pred:.1f}ms")
        print(f"\n{classification_report(y_te, y_pred, target_names=label_names)}")

        records.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "F1 (weighted)": round(f1w, 4),
            "F1 (macro)": round(f1m, 4),
            "Train (s)": round(t_train, 2),
            "Predict (ms)": round(t_pred, 1),
        })
        trained[name] = (pipe, y_pred)

    df_res = pd.DataFrame(records).sort_values("F1 (weighted)", ascending=False)
    return df_res, trained


def cross_validate_models(texts_clean: list, y: np.ndarray, pipelines: dict,
                          cv: int = 5, random_state: int = 42) -> pd.DataFrame:
    """Stratified K-Fold cross-validation: báo cáo Accuracy/F1 mean ± std."""
    print("\n" + "=" * 65)
    print("🔁 CROSS-VALIDATION (Stratified %d-Fold)" % cv)
    print("=" * 65)

    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=random_state)
    rows = []
    for name, pipe in pipelines.items():
        acc = cross_val_score(pipe, texts_clean, y, cv=skf,
                              scoring="accuracy", n_jobs=-1)
        f1w = cross_val_score(pipe, texts_clean, y, cv=skf,
                              scoring="f1_weighted", n_jobs=-1)
        f1m = cross_val_score(pipe, texts_clean, y, cv=skf,
                              scoring="f1_macro", n_jobs=-1)

        print(f"\n▶ {name}")
        print(f"   Accuracy   : {acc.mean():.4f} ± {acc.std():.4f}")
        print(f"   F1 weighted: {f1w.mean():.4f} ± {f1w.std():.4f}")
        print(f"   F1 macro   : {f1m.mean():.4f} ± {f1m.std():.4f}")

        rows.append({
            "Model": name,
            "Accuracy (mean ± std)": f"{acc.mean():.4f} ± {acc.std():.4f}",
            "F1-weighted (mean ± std)": f"{f1w.mean():.4f} ± {f1w.std():.4f}",
            "F1-macro (mean ± std)": f"{f1m.mean():.4f} ± {f1m.std():.4f}",
        })

    df_cv = pd.DataFrame(rows)
    df_cv.to_csv(TABLES_DIR / "cross_validation_results.csv", index=False)
    print("\n✅ Saved → cross_validation_results.csv")
    return df_cv


def grid_search_tuning(X_tr, y_tr, cv: int = 3) -> dict:
    """GridSearchCV: tìm hyperparameter tốt nhất cho từng mô hình."""
    print("\n" + "=" * 65)
    print("🔧 HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 65)

    tfidf_params = dict(ngram_range=(1, 2), max_features=50_000,
                        sublinear_tf=True, min_df=2)

    def _pipe(clf):
        return Pipeline([("vec", TfidfVectorizer(**tfidf_params)), ("clf", clf)])

    base = {
        "MultinomialNB": _pipe(MultinomialNB()),
        "ComplementNB": _pipe(ComplementNB()),
        "LinearSVC": _pipe(LinearSVC(max_iter=2000, random_state=42)),
        "SVC (RBF)": _pipe(SVC(kernel="rbf", random_state=42)),
    }
    param_grids = {
        "MultinomialNB": {"clf__alpha": [0.01, 0.1, 0.5, 1.0]},
        "ComplementNB": {"clf__alpha": [0.1, 0.5, 1.0, 2.0]},
        "LinearSVC": {"clf__C": [0.1, 1.0, 10.0]},
        "SVC (RBF)": {"clf__C": [0.1, 1.0, 10.0], "clf__gamma": ["scale", "auto"]},
    }

    best_params, rows = {}, []
    for name, grid in param_grids.items():
        gs = GridSearchCV(base[name], grid, cv=cv, scoring="f1_weighted",
                          n_jobs=-1, verbose=0)
        gs.fit(X_tr, y_tr)
        best_params[name] = gs.best_params_
        rows.append({
            "Model": name,
            "Best Params": str(gs.best_params_),
            "CV F1 (weighted)": round(gs.best_score_, 4),
        })
        print(f"\n▶ {name}")
        print(f"   Best params : {gs.best_params_}")
        print(f"   Best CV F1  : {gs.best_score_:.4f}")

    df_gs = pd.DataFrame(rows)
    df_gs.to_csv(TABLES_DIR / "hyperparameter_tuning.csv", index=False)
    print("\n✅ Saved → hyperparameter_tuning.csv")
    return best_params


# ════════════════════════════════════════════════════════════
# 5. VISUALIZATIONS
# ════════════════════════════════════════════════════════════

def plot_eda(texts: list, y: np.ndarray, label_names: list):
    df = pd.DataFrame({"text": texts, "label": [label_names[i] for i in y]})
    df["length"] = df["text"].str.split().str.len()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("EDA — Sentiment Dataset", fontsize=14, fontweight="bold")

    # Label distribution
    counts = df["label"].value_counts()
    colors = ["#F44336", "#9E9E9E", "#4CAF50"] if len(label_names) == 3 \
             else ["#F44336", "#4CAF50"]
    axes[0].bar(counts.index, counts.values, color=colors[:len(counts)])
    axes[0].set_title("Phân phối nhãn")
    axes[0].set_ylabel("Số lượng")
    for i, (label, val) in enumerate(counts.items()):
        axes[0].text(i, val + 100, f"{val:,}", ha="center", fontsize=10)

    # Text length by class
    for label in label_names:
        subset = df[df["label"] == label]["length"]
        axes[1].hist(subset, bins=40, alpha=0.6, label=label)
    axes[1].set_title("Độ dài text theo nhãn")
    axes[1].set_xlabel("Số từ sau tiền xử lý")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "01_eda.png", dpi=130, bbox_inches="tight")
    print("📊 Saved → 01_eda.png")
    plt.close()


def plot_model_comparison(df_res: pd.DataFrame):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("So sánh mô hình — Sentiment Analysis", fontsize=14, fontweight="bold")
    metrics = ["Accuracy", "F1 (weighted)", "F1 (macro)"]
    palette = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]

    for ax, metric in zip(axes, metrics):
        vals   = df_res[metric].values
        models = df_res["Model"].values
        bars   = ax.barh(models, vals, color=palette[:len(models)])
        ax.set_xlim(max(0, vals.min() - 0.08), 1.01)
        ax.set_title(metric, fontsize=12)
        ax.axvline(x=vals.max(), color="red", linestyle="--", alpha=0.4)
        for bar, v in zip(bars, vals):
            ax.text(v + 0.003, bar.get_y() + bar.get_height()/2,
                    f"{v:.4f}", va="center", fontsize=9)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "02_model_comparison.png", dpi=130, bbox_inches="tight")
    print("📊 Saved → 02_model_comparison.png")
    plt.close()


def plot_confusion_matrices(y_te, trained: dict, label_names: list):
    n   = len(trained)
    fig, axes = plt.subplots(1, n, figsize=(5*n, 4))
    if n == 1:
        axes = [axes]
    fig.suptitle("Confusion Matrices — SVM vs Naive Bayes", fontsize=13, fontweight="bold")

    for ax, (name, (_, y_pred)) in zip(axes, trained.items()):
        cm = confusion_matrix(y_te, y_pred)
        sns.heatmap(cm, annot=True, fmt="d", ax=ax, cmap="Blues",
                    xticklabels=label_names, yticklabels=label_names,
                    linewidths=0.5, linecolor="white")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "03_confusion_matrices.png", dpi=130, bbox_inches="tight")
    print("📊 Saved → 03_confusion_matrices.png")
    plt.close()


def plot_top_features(trained: dict, label_names: list, top_n: int = 15):
    """Vẽ top features của LinearSVC và ComplementNB."""
    fig, axes = plt.subplots(1, len(label_names),
                             figsize=(6 * len(label_names), 4))
    if len(label_names) == 1:
        axes = [axes]

    # Ưu tiên LinearSVC
    pipe_name = "LinearSVC" if "LinearSVC" in trained else list(trained.keys())[0]
    pipe, _   = trained[pipe_name]
    vec       = pipe.named_steps["vec"]
    clf       = pipe.named_steps["clf"]
    feat      = vec.get_feature_names_out()

    fig.suptitle(f"Top {top_n} Features — {pipe_name}", fontsize=13, fontweight="bold")

    if hasattr(clf, "coef_"):
        coef = clf.coef_ if len(label_names) > 2 else np.vstack([-clf.coef_[0], clf.coef_[0]])
        for ax, (i, label) in zip(axes, enumerate(label_names)):
            idx  = np.argsort(coef[i])[-top_n:][::-1]
            vals = coef[i][idx]
            ax.barh(feat[idx][::-1], vals[::-1], color="#1976D2")
            ax.set_title(f"[{label}]")
            ax.set_xlabel("Coefficient")

    elif hasattr(clf, "feature_log_prob_"):
        for ax, (i, label) in zip(axes, enumerate(label_names)):
            idx  = np.argsort(clf.feature_log_prob_[i])[-top_n:][::-1]
            vals = clf.feature_log_prob_[i][idx]
            ax.barh(feat[idx][::-1], vals[::-1], color="#388E3C")
            ax.set_title(f"[{label}]")
            ax.set_xlabel("Log Probability")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "04_top_features.png", dpi=130, bbox_inches="tight")
    print("📊 Saved → 04_top_features.png")
    plt.close()


def plot_speed_comparison(df_res: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, 4))
    models = df_res["Model"].values
    times  = df_res["Train (s)"].values
    palette = ["#1976D2", "#388E3C", "#F57C00", "#7B1FA2"]
    bars = ax.bar(models, times, color=palette[:len(models)])
    ax.set_title("Thời gian Train (giây) — Tốc độ NB vs SVM", fontsize=12)
    ax.set_ylabel("Giây")
    for bar, t in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f"{t:.2f}s", ha="center", fontsize=10)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "05_speed_comparison.png", dpi=130, bbox_inches="tight")
    print("📊 Saved → 05_speed_comparison.png")
    plt.close()


def plot_wordcloud(texts_clean: list, y: np.ndarray, label_names: list,
                   max_words: int = 100):
    """Word cloud các từ đặc trưng theo từng lớp sentiment."""
    try:
        from wordcloud import WordCloud
    except ImportError:
        print("⚠️  wordcloud chưa cài — bỏ qua plot_wordcloud.")
        return

    n = len(label_names)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]
    fig.suptitle("Word Cloud — Từ đặc trưng theo lớp", fontsize=14, fontweight="bold")

    for ax, (i, label) in zip(axes, enumerate(label_names)):
        subset = " ".join(t for t, yy in zip(texts_clean, y) if yy == i)
        if not subset.strip():
            subset = "no data"
        wc = WordCloud(width=800, height=500, max_words=max_words,
                       background_color="white", random_state=42).generate(subset)
        ax.imshow(wc, interpolation="bilinear")
        ax.set_title(f"[{label}]", fontsize=12)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "06_wordcloud.png", dpi=130, bbox_inches="tight")
    print("📊 Saved → 06_wordcloud.png")
    plt.close()


# ════════════════════════════════════════════════════════════
# 6. INFERENCE
# ════════════════════════════════════════════════════════════

def demo_inference(pipe: Pipeline, label_names: list, lang: str = "en"):
    SAMPLES = {
        "en": [
            ("Positive", "This product is absolutely amazing, best purchase ever!"),
            ("Negative", "Terrible quality, broke after one day. Very disappointed."),
            ("Negative?", "not bad at all, actually quite decent"),   # NB có thể sai
            ("Positive?", "Oh great, another bug in this software..."),  # Sarcasm
            ("Neutral",   "The product arrived on time. Nothing special."),
        ],
        "vi": [
            ("Positive", "Sản phẩm rất tốt, chất lượng vượt kỳ vọng!"),
            ("Negative", "Hàng kém chất lượng, không đúng mô tả, thất vọng."),
            ("Neutral",  "Sản phẩm bình thường, giao hàng đúng hẹn."),
        ],
    }
    samples = SAMPLES.get(lang, SAMPLES["en"])

    print("\n" + "=" * 65)
    print("🔮 DEMO INFERENCE")
    print("=" * 65)
    print(f"{'Expected':10s} | {'Predicted':10s} | Text")
    print("-" * 65)
    for expected, text in samples:
        cleaned = [clean_vi(text) if lang == "vi" else clean_en(text)]
        pred    = label_names[pipe.predict(cleaned)[0]]
        mark    = "✅" if pred.lower() in expected.lower() or "?" in expected else "❌"
        print(f"{expected:10s} | {pred:10s} {mark} | {text[:50]}")


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Sentiment Analysis: SVM vs Naive Bayes"
    )
    parser.add_argument("--source", choices=["imdb", "amazon", "uci", "vn"], default="imdb",
                        help="Dataset loader to use; uci expects the prepared UCI CSV")
    parser.add_argument("--data",   type=str, default=None,
                        help="Path to dataset CSV/txt")
    parser.add_argument("--demo",   action="store_true",
                        help="Chạy với dữ liệu mẫu, không cần file")
    parser.add_argument("--sample", type=int, default=None,
                        help="Giới hạn số mẫu (ví dụ: 10000)")
    parser.add_argument("--output-dir", type=str, default="outputs",
                        help="Directory for figures and result tables")
    args = parser.parse_args()

    configure_output_dirs(args.output_dir)

    # ── 1. Load ──
    if args.demo:
        texts, y, label_names = load_demo()
        lang = "en"
    elif args.source == "imdb":
        texts, y, label_names = load_imdb(args.data, sample=args.sample)
        lang = "en"
    elif args.source == "amazon":
        texts, y, label_names = load_amazon(args.data, sample=args.sample or 100_000)
        lang = "en"
    elif args.source == "uci":
        texts, y, label_names = load_uci(args.data, sample=args.sample)
        lang = "en"
    elif args.source == "vn":
        texts, y, label_names = load_vietnamese(args.data, sample=args.sample)
        lang = "vi"

    print(f"🏷  Labels ({len(label_names)}): {label_names}")

    # ── 2. EDA plot ──
    plot_eda(texts, y, label_names)

    # ── 3. Preprocess ──
    texts_clean = preprocess(texts, lang=lang)

    # ── 4. Split ──
    X_tr, X_te, y_tr, y_te = train_test_split(
        texts_clean, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"📦 Train: {len(X_tr):,} | Test: {len(X_te):,}\n")

    # ── 5. Build & Train ──
    pipelines = build_pipelines()
    df_res, trained = train_compare(X_tr, X_te, y_tr, y_te, pipelines, label_names)

    # ── 6. Kết quả tổng hợp ──
    print("\n" + "=" * 65)
    print("📈 KẾT QUẢ SO SÁNH TỔNG HỢP")
    print("=" * 65)
    print(df_res.to_string(index=False))
    df_res.to_csv(TABLES_DIR / "comparison_results.csv", index=False)
    print("\n✅ Saved → comparison_results.csv")

    # ── 6b. Cross-validation & Hyperparameter tuning ──
    cross_validate_models(texts_clean, y, pipelines)
    grid_search_tuning(X_tr, y_tr)

    # ── 7. Biểu đồ ──
    plot_model_comparison(df_res)
    plot_confusion_matrices(y_te, trained, label_names)
    plot_top_features(trained, label_names)
    plot_speed_comparison(df_res)
    plot_wordcloud(texts_clean, y, label_names)

    # ── 8. Inference demo ──
    best_name   = df_res.iloc[0]["Model"]
    best_pipe, _ = trained[best_name]
    demo_inference(best_pipe, label_names, lang=lang)

    print("\n" + "=" * 65)
    print(f"🏆 Best model: {best_name} (F1={df_res.iloc[0]['F1 (weighted)']:.4f})")
    print("=" * 65)
    print("📁 Output files:")
    for f in [FIGURES_DIR / "01_eda.png", FIGURES_DIR / "02_model_comparison.png",
              FIGURES_DIR / "03_confusion_matrices.png", FIGURES_DIR / "04_top_features.png",
              FIGURES_DIR / "05_speed_comparison.png", FIGURES_DIR / "06_wordcloud.png",
              TABLES_DIR / "comparison_results.csv",
              TABLES_DIR / "cross_validation_results.csv",
              TABLES_DIR / "hyperparameter_tuning.csv"]:
        print(f"   → {f}")


if __name__ == "__main__":
    main()
