import re
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

INPUT_FILE = sys.argv[1] if len(sys.argv) > 1 else "email.csv"
OUTPUT_FILE = "email_classified.csv"
LONG_THRESHOLD = 100  # characters
K_NEIGHBORS = 5       # how many nearest messages get a vote

SPAM_KEYWORDS = [
    "free", "winner", "prize", "won", "win", "cash", "reward", "offer",
    "urgent", "congratulations", "congrats", "claim", "guaranteed", "award",
    "click", "act now", "limited time", "bonus", "credit", "loan",
    "discount", "deal", "txt", "text", "reply", "stop", "call now", "mobile",
]
SPECIAL_CHARS = set("$£€%#*!@&^")

KEYWORD_RE = re.compile(
    r"\b(?:" + "|".join(re.escape(k) for k in SPAM_KEYWORDS) + r")\b", re.IGNORECASE
)
URL_RE = re.compile(
    r"(?:https?://\S+|www\.\S+|\b[\w\-]+\.(?:com|net|org|biz|info|tv|co\.uk|uk)\b)",
    re.IGNORECASE,
)
DOMAIN_RE = re.compile(
    r"(?:@|www\.|https?://)([\w\-]+(?:\.[\w\-]+)*\.[a-z]{2,})", re.IGNORECASE
)
CAPS_WORD_RE = re.compile(r"\b[A-Z]{3,}\b")
LONG_NUMBER_RE = re.compile(r"\d{3,}")

FEATURE_NAMES = [
    "caps_ratio", "caps_words", "keyword_count", "url_count", "special_count",
    "length", "is_long", "domain_mismatch", "has_long_number", "digit_count",
]


def extract_features(text):
    text = str(text)
    letters = [c for c in text if c.isalpha()]
    caps_ratio = sum(c.isupper() for c in letters) / max(len(letters), 1)
    # Strip a leading "www." so www.site.com and info@site.com count as one domain
    domains = {re.sub(r"^www\.", "", d.lower()) for d in DOMAIN_RE.findall(text)}
    return [
        caps_ratio,
        len(CAPS_WORD_RE.findall(text)),
        len({m.lower() for m in KEYWORD_RE.findall(text)}),
        len(URL_RE.findall(text)),
        sum(c in SPECIAL_CHARS for c in text),
        len(text),
        int(len(text) > LONG_THRESHOLD),
        int(len(domains) > 1),
        int(bool(LONG_NUMBER_RE.search(text))),
        sum(c.isdigit() for c in text),
    ]


def predict_email(model, text):
    """Classify one new message with a fitted model."""
    x = pd.DataFrame([extract_features(text)], columns=FEATURE_NAMES)
    return "spam" if model.predict(x)[0] == 1 else "ham"


def main():
    df = pd.read_csv(INPUT_FILE)
    df["Category"] = df["Category"].astype(str).str.strip().str.lower()
    bad = ~df["Category"].isin(["spam", "ham"])
    if bad.any():
        print(f"Dropping {bad.sum()} row(s) that are not labelled spam/ham")
        df = df[~bad].reset_index(drop=True)
    y = (df["Category"] == "spam").astype(int).to_numpy()

    X = pd.DataFrame(
        [extract_features(t) for t in df["Message"]], columns=FEATURE_NAMES
    )
    print(f"{len(df)} messages: {y.sum()} spam, {(y == 0).sum()} ham\n")

    model = make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=K_NEIGHBORS)
    )

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    pred = cross_val_predict(model, X, y, cv=cv, groups=df["Message"].astype(str))

    print(f"kNN (k = {K_NEIGHBORS}) accuracy: {np.mean(pred == y) * 100:.2f}%\n")
    print(classification_report(y, pred, target_names=["Ham", "Spam"], digits=3))

    cm = confusion_matrix(y, pred)
    print("Confusion matrix:")
    print(f"  True Negatives  (ham  -> ham ): {cm[0, 0]}")
    print(f"  False Positives (ham  -> spam): {cm[0, 1]}")
    print(f"  False Negatives (spam -> ham ): {cm[1, 0]}")
    print(f"  True Positives  (spam -> spam): {cm[1, 1]}")

    final_model = model.fit(X, y)

    out = pd.DataFrame({
        "Correct": np.where(pred == y, "T", "F"),
        "Actual": df["Category"],
        "Predicted": np.where(pred == 1, "spam", "ham"),
    })
    out = pd.concat([out, X.round(3), df[["Message"]]], axis=1)
    out.to_csv(OUTPUT_FILE, index=False)

    print(f"\nWrote {OUTPUT_FILE}")
    print(f"Total classified: {len(out)}")
    print(f"Correct:   {(out['Correct'] == 'T').sum()}")
    print(f"Incorrect: {(out['Correct'] == 'F').sum()}")

if __name__ == "__main__":
    main()
