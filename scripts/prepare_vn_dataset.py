"""Tải UIT Vietnamese Students' Feedback (UIT-VSFC) về data/raw/vn_feedback.csv.

Dataset: Nguyen et al. (2018) — UIT-VSFC: Vietnamese Students' Feedback Corpus.
Nguồn gốc file (Google Drive) lấy từ loader trên HuggingFace:
https://huggingface.co/datasets/uitnlp/vietnamese_students_feedback

Nhãn: 0 = NEGATIVE, 1 = NEUTRAL, 2 = POSITIVE (3 lớp).
Output CSV gồm 2 cột: `text`, `sentiment` (giá trị: negative/neutral/positive).

Cách chạy:
    .\\.venv\\Scripts\\python.exe scripts\\prepare_vn_dataset.py
"""
from pathlib import Path

import pandas as pd
import requests

OUT_PATH = Path("data") / "raw" / "vn_feedback.csv"

# URL gốc (Google Drive) — sentences (text) và sentiments (nhãn 0/1/2).
_SENTENCES = {
    "train": "https://drive.google.com/uc?id=1nzak5OkrheRV1ltOGCXkT671bmjODLhP&export=download",
    "validation": "https://drive.google.com/uc?id=1sMJSR3oRfPc3fe1gK-V3W5F24tov_517&export=download",
    "test": "https://drive.google.com/uc?id=1aNMOeZZbNwSRkjyCWAGtNCMa3YrshR-n&export=download",
}
_SENTIMENTS = {
    "train": "https://drive.google.com/uc?id=1ye-gOZIBqXdKOoi_YxvpT6FeRNmViPPv&export=download",
    "validation": "https://drive.google.com/uc?id=1GiY1AOp41dLXIIkgES4422AuDwmbUseL&export=download",
    "test": "https://drive.google.com/uc?id=1vkQS5gI0is4ACU58-AbWusnemw7KZNfO&export=download",
}

LABEL_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def _download_text(url: str) -> str:
    """Tải file text từ Google Drive, trả về nội dung dạng chuỗi."""
    resp = requests.get(url, timeout=180)
    resp.raise_for_status()
    head = resp.text[:600].lower()
    if "<html" in head or "doctype" in head:
        raise RuntimeError(
            "Google Drive trả về trang HTML thay vì file text "
            "(file lớn cần xác nhận tải — hãy tải thủ công)."
        )
    return resp.text


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ("train", "validation", "test"):
        sentences = _download_text(_SENTENCES[split]).splitlines()
        sentiments = _download_text(_SENTIMENTS[split]).splitlines()
        if len(sentences) != len(sentiments):
            raise RuntimeError(f"{split}: độ dài lệch ({len(sentences)} vs {len(sentiments)})")

        for sentence, label in zip(sentences, sentiments):
            sentence = sentence.strip()
            if sentence:
                rows.append({
                    "text": sentence,
                    "sentiment": LABEL_MAP[int(label.strip())],
                })
        print(f"  {split}: {len(sentences):,} câu")

    df = pd.DataFrame(rows).drop_duplicates().reset_index(drop=True)
    print(f"Tổng cộng (sau drop_duplicates): {len(df):,} rows")
    print(f"Phân phối nhãn:\n{df['sentiment'].value_counts().to_string()}")

    df.to_csv(OUT_PATH, index=False)
    print(f"✅ Saved → {OUT_PATH}")


if __name__ == "__main__":
    main()
