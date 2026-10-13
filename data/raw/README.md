# Real Dataset: UCI Sentiment Labelled Sentences

Download the dataset from the UCI Machine Learning Repository:

https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences

The official UCI collection contains 3,000 labelled sentences: 500 positive
and 500 negative examples from each of IMDb, Amazon, and Yelp. Dataset DOI:
`10.24432/C57604`. Cite Kotzias et al., *From Group to Individual Labels
Using Deep Features*, KDD 2015, when using the dataset.

Prepare `uci_sentiment_reviews.csv` in this folder with these columns:

```text
text,sentiment,source
"The service was excellent.",positive,yelp
"The product stopped working.",negative,amazon
```

Accepted sentiment values are `positive` and `negative`. The notebook expects
reviews from the `imdb`, `amazon`, and `yelp` sources.

## Vietnamese dataset: UIT-VSFC

`vn_feedback.csv` is the UIT Vietnamese Students' Feedback corpus (3 classes
`negative / neutral / positive`). Generate it with:

```powershell
.\.venv\Scripts\python.exe .\scripts\prepare_vn_dataset.py
```

Cite Nguyen et al., *UIT-VSFC: Vietnamese Students' Feedback Corpus for
Sentiment Analysis*, KSE 2018, DOI: `10.1109/KSE.2018.8573337`.
