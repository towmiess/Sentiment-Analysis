# Real Data Results

## Dataset and method

The analysis uses the UCI Sentiment Labelled Sentences collection, containing
3,000 English review sentences from IMDb, Amazon, and Yelp. The target is binary
sentiment (`negative` or `positive`), with 1,500 examples in each class. A
stratified 80/20 split with `random_state=42` gives 2,400 training examples and
600 test examples. Text is processed with the project's English cleaning,
stopword, and stemming steps, then vectorized with TF-IDF.

## Model comparison

| Model | Family | Accuracy | Weighted F1 | Macro F1 | Train (s) | Predict (ms) |
|---|---|---:|---:|---:|---:|---:|
| SVC (RBF) | SVM | 0.8283 | 0.8283 | 0.8283 | 0.62 | 121.5 |
| ComplementNB | Naive Bayes | 0.8233 | 0.8233 | 0.8233 | 0.07 | 9.5 |
| LinearSVC | SVM | 0.8233 | 0.8233 | 0.8233 | 0.09 | 10.0 |
| MultinomialNB | Naive Bayes | 0.8050 | 0.8050 | 0.8050 | 0.12 | 13.6 |

On this split, SVC (RBF) has the highest F1 by 0.5 percentage points over
ComplementNB and LinearSVC. It also has substantially higher measured prediction
latency. For a low-latency baseline, LinearSVC and ComplementNB are competitive.

## Cross-validation (Stratified 5-Fold)

To avoid relying on a single split, a stratified 5-fold cross-validation was run
(`shuffle=True`, `random_state=42`). Mean ± standard deviation across folds:

| Model | Accuracy | Weighted F1 | Macro F1 |
|---|---:|---:|---:|
| MultinomialNB | 0.8250 ± 0.0091 | 0.8250 ± 0.0091 | 0.8250 ± 0.0091 |
| ComplementNB | 0.8293 ± 0.0065 | 0.8293 ± 0.0064 | 0.8293 ± 0.0064 |
| LinearSVC | 0.8337 ± 0.0046 | 0.8336 ± 0.0046 | 0.8336 ± 0.0046 |
| SVC (RBF) | 0.8383 ± 0.0072 | 0.8381 ± 0.0072 | 0.8381 ± 0.0072 |

The ranking is stable and the standard deviations are small (≤ 0.01), so the
conclusion does not depend on a lucky split.

## Hyperparameter tuning

`GridSearchCV` (3-fold, `f1_weighted`) selected these parameters:

| Model | Best params | CV F1 (weighted) |
|---|---:|---:|
| MultinomialNB | alpha=0.5 | 0.8153 |
| ComplementNB | alpha=2.0 | 0.8199 |
| LinearSVC | C=0.1 | 0.8236 |
| SVC (RBF) | C=1.0, gamma=scale | 0.8199 |

The tuned values are close to the defaults, so the main comparison above is
essentially unchanged.

## Interpretation limits

The hold-out scores describe one fixed 80/20 split; the cross-validation section
above provides a more robust estimate. The dataset is small and balanced, and the
source domains differ, but the split is stratified by sentiment only; it does not
test generalization to an entirely unseen review source. Timing is a single local
measurement and varies with the machine and system load.

The confusion-matrix counts are available in
`outputs/tables/uci_confusion_matrices.csv`; the per-class metrics are in
`outputs/tables/uci_classification_report.csv`.

For binary labels, rows below use TN, FP, FN, TP order:

| Model | TN | FP | FN | TP |
|---|---:|---:|---:|---:|
| SVC (RBF) | 255 | 45 | 58 | 242 |
| ComplementNB | 249 | 51 | 55 | 245 |
| LinearSVC | 247 | 53 | 53 | 247 |
| MultinomialNB | 246 | 54 | 63 | 237 |

## Vietnamese experiment (UIT-VSFC)

The pipeline was also run on UIT Vietnamese Students' Feedback (16,175 sentences,
3 classes: negative / neutral / positive), sampled to 6,000 examples
(`random_state=42`) to keep runtime reasonable. Preprocessing uses
`underthesea.word_tokenize`. Results on an 80/20 stratified split:

| Model | Family | Accuracy | Weighted F1 | Macro F1 | Train (s) | Predict (ms) |
|---|---|---:|---:|---:|---:|---:|
| LinearSVC | SVM | 0.8942 | 0.8866 | 0.6976 | 0.29 | 40.1 |
| SVC (RBF) | SVM | 0.8925 | 0.8804 | 0.6667 | 6.0 | 1031.7 |
| MultinomialNB | Naive Bayes | 0.8758 | 0.8654 | 0.6639 | 0.33 | 57.8 |
| ComplementNB | Naive Bayes | 0.8742 | 0.8647 | 0.6594 | 0.31 | 54.2 |

`LinearSVC` leads here (weighted F1 0.8866), ahead of `SVC (RBF)`. Macro F1 is
much lower (≈ 0.66–0.70) because the `neutral` class is only ~4% of the data.
Results are saved under `outputs_vn/`. Download the data with
`scripts/prepare_vn_dataset.py`.

## Reproduce

```powershell
.\.venv\Scripts\python.exe .\sentiment_svm_nb.py --source uci --data .\data\raw\uci_sentiment_reviews.csv --output-dir .\outputs
```

The detailed notebook is `notebooks/sentiment_analysis.ipynb`. Result tables
and figures are saved under `outputs/tables/` and `outputs/figures/`.

## Citation

Kotzias, D., Denil, M., de Freitas, N., & Smyth, P. (2015). *From Group to
Individual Labels Using Deep Features*. Proceedings of the 21st ACM SIGKDD
International Conference on Knowledge Discovery and Data Mining. UCI Machine
Learning Repository, Sentiment Labelled Sentences, DOI: 10.24432/C57604.
