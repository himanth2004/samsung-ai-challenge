import os
import re
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_texts():
    human_path = os.path.join(BASE_DIR, "human_text.txt")
    robot_path = os.path.join(BASE_DIR, "robot_text.txt")

    with open(human_path, "r", encoding="utf-8", errors="ignore") as f:
        human_texts = [line.strip() for line in f if line.strip()]

    with open(robot_path, "r", encoding="utf-8", errors="ignore") as f:
        robot_texts = [line.strip() for line in f if line.strip()]

    texts = human_texts + robot_texts
    labels = [0] * len(human_texts) + [1] * len(robot_texts)
    return texts, labels


def build_cleaner():
    replacements = {
        "he's": "he is", "she's": "she is", "it's": "it is", "I'm": "I am",
        "you're": "you are", "we're": "we are", "they're": "they are",
        "he'll": "he will", "she'll": "she will", "it'll": "it will",
        "i'll": "i will", "you'll": "you will", "we'll": "we will",
        "they'll": "they will", "he'd": "he would", "she'd": "she would",
        "it'd": "it would", "i'd": "i would", "you'd": "you would",
        "we'd": "we would", "they'd": "they would", "haven't": "have not",
        "hasn't": "has not", "hadn't": "had not", "don't": "do not",
        "doesn't": "does not", "didn't": "did not", "can't": "cannot",
        "won't": "will not", "wouldn't": "would not", "shouldn't": "should not",
        "mightn't": "might not", "mustn't": "must not", "aren't": "are not",
        "isn't": "is not", "wasn't": "was not", "weren't": "were not",
        "im": "i am", "u": "you"
    }

    def clean_text(text: str) -> str:
        text = text.lower()
        for k, v in replacements.items():
            text = text.replace(k, v)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        return text.strip()

    return clean_text


def main():
    texts, labels = load_texts()
    clean_text = build_cleaner()
    texts = [clean_text(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english", max_features=5000, ngram_range=(1, 2))),
        ("clf", LogisticRegression(max_iter=2000, n_jobs=None, random_state=42))
    ])

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred, digits=4))

    out_path = os.path.join(BASE_DIR, "typing_pipeline.pkl")
    joblib.dump(pipeline, out_path)
    print(f"Saved typing pipeline to: {out_path}")


if __name__ == "__main__":
    main()


