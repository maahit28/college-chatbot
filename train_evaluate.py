"""
train_evaluate.py
─────────────────
Standalone script for training, evaluating, and comparing
Naive Bayes and Logistic Regression models for the HP Chatbot.

Run: python train_evaluate.py
"""

import re
import string
import random
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

# ── Simple stopwords (no NLTK dependency needed for this script) ──
STOPWORDS = {
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "its", "they", "what", "which", "who", "this", "that", "is", "are",
    "was", "were", "be", "been", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "a", "an",
    "the", "and", "or", "but", "in", "on", "at", "to", "for", "of",
    "with", "by", "from", "about", "as", "into", "can", "am",
}

# ── Knowledge base ───────────────────────────────────────────────
INTENTS_DATA = {
    "greetings": [
        "hello", "hi", "hey", "good morning", "good afternoon", "howdy",
        "greetings", "namaste", "hi there", "what's up", "hiya", "good day",
    ],
    "fees": [
        "what is the fee", "fees structure", "how much fees", "annual fees",
        "semester fees", "tuition fee", "total fees", "fee details",
        "how much does it cost", "fee per year", "fee amount", "charges",
        "what does it cost", "total expenditure",
    ],
    "courses": [
        "courses available", "what courses", "which programmes", "branches offered",
        "engineering courses", "what can i study", "list of courses", "what do you offer",
        "available programmes", "departments", "subjects offered", "programmes list",
        "what branches", "available branches",
    ],
    "hostel": [
        "hostel facility", "accommodation", "hostel available", "hostel charges",
        "hostel rooms", "boarding facility", "where will i stay", "campus accommodation",
        "hostel fee", "is hostel available", "dormitory", "residential facility",
        "room and board", "hostel details",
    ],
    "placements": [
        "placement record", "how are placements", "highest package", "companies visit",
        "placement percentage", "average salary", "job opportunities",
        "campus recruitment", "placement stats", "average package",
        "who recruits", "salary package", "campus placements", "job prospects",
    ],
    "contact": [
        "contact number", "phone number", "email address", "how to contact",
        "admission office", "address", "website", "helpline",
        "get in touch", "reach them", "contact info", "phone", "email",
    ],
    "exit": [
        "bye", "goodbye", "see you", "thanks", "thank you", "exit", "quit", "done",
        "that's all", "no more questions", "see ya", "take care", "farewell",
    ],
}


# ═══════════════════════════════════════════════════════════════════
#  PREPROCESSING
# ═══════════════════════════════════════════════════════════════════
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[" + string.punctuation + r"]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


# ═══════════════════════════════════════════════════════════════════
#  BUILD DATASET
# ═══════════════════════════════════════════════════════════════════
def build_dataset():
    X, y = [], []
    for intent, patterns in INTENTS_DATA.items():
        for pattern in patterns:
            X.append(preprocess(pattern))
            y.append(intent)
    return X, y


# ═══════════════════════════════════════════════════════════════════
#  TRAIN AND EVALUATE
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 60)
    print("  HP College Chatbot – Model Training & Evaluation")
    print("=" * 60)

    # Build dataset
    X_raw, y = build_dataset()
    print(f"\n📂 Dataset size: {len(X_raw)} samples across {len(set(y))} intents")
    print(f"   Intents: {sorted(set(y))}\n")

    # ── Vectorization ──────────────────────────────────────────────
    print("🔧 Feature Extraction: CountVectorizer (Bag of Words)")
    vectorizer = CountVectorizer(ngram_range=(1, 2))
    X = vectorizer.fit_transform(X_raw)
    print(f"   Vocabulary size: {len(vectorizer.vocabulary_)} features\n")

    # ── Train/Test Split ───────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    print(f"📊 Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}\n")

    # ── Naive Bayes ────────────────────────────────────────────────
    print("─" * 40)
    print("MODEL 1: Multinomial Naive Bayes")
    nb = MultinomialNB(alpha=0.5)
    nb.fit(X_train, y_train)
    nb_pred = nb.predict(X_test)
    nb_acc  = accuracy_score(y_test, nb_pred)
    nb_cv   = cross_val_score(nb, X, y, cv=5).mean()
    print(f"  Test Accuracy  : {nb_acc * 100:.2f}%")
    print(f"  5-Fold CV Acc  : {nb_cv * 100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, nb_pred, zero_division=0))

    # ── Logistic Regression ────────────────────────────────────────
    print("─" * 40)
    print("MODEL 2: Logistic Regression")
    lr = LogisticRegression(max_iter=500, C=1.5, solver="lbfgs", multi_class="auto")
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)
    lr_acc  = accuracy_score(y_test, lr_pred)
    lr_cv   = cross_val_score(lr, X, y, cv=5).mean()
    print(f"  Test Accuracy  : {lr_acc * 100:.2f}%")
    print(f"  5-Fold CV Acc  : {lr_cv * 100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, lr_pred, zero_division=0))

    # ── Summary ────────────────────────────────────────────────────
    print("─" * 40)
    print("COMPARISON SUMMARY")
    print(f"  Naive Bayes        : {nb_acc * 100:.2f}% (CV: {nb_cv * 100:.2f}%)")
    print(f"  Logistic Regression: {lr_acc * 100:.2f}% (CV: {lr_cv * 100:.2f}%)")
    best = "Logistic Regression" if lr_acc >= nb_acc else "Naive Bayes"
    print(f"\n  ✅ Best Model: {best}")

    # ── Plots ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    fig.suptitle("HP Chatbot – Model Evaluation", fontsize=14, fontweight="bold")

    # Bar chart
    axes[0].bar(
        ["Naive Bayes", "Logistic Regression"],
        [nb_acc * 100, lr_acc * 100],
        color=["#4a86b8", "#1d4e89"],
        width=0.45,
        edgecolor="none",
    )
    axes[0].set_ylim([70, 105])
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].set_title("Test Accuracy Comparison")
    for i, v in enumerate([nb_acc * 100, lr_acc * 100]):
        axes[0].text(i, v + 0.5, f"{v:.1f}%", ha="center", fontweight="bold")
    axes[0].spines[["top", "right"]].set_visible(False)

    # NB Confusion Matrix
    labels = sorted(set(y))
    cm_nb = confusion_matrix(y_test, nb_pred, labels=labels)
    axes[1].imshow(cm_nb, cmap="Blues")
    axes[1].set_xticks(range(len(labels)))
    axes[1].set_yticks(range(len(labels)))
    axes[1].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[1].set_yticklabels(labels, fontsize=8)
    axes[1].set_title("NB Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[1].text(j, i, cm_nb[i, j], ha="center", va="center",
                         color="white" if cm_nb[i, j] > cm_nb.max() / 2 else "black", fontsize=8)

    # LR Confusion Matrix
    cm_lr = confusion_matrix(y_test, lr_pred, labels=labels)
    axes[2].imshow(cm_lr, cmap="Blues")
    axes[2].set_xticks(range(len(labels)))
    axes[2].set_yticks(range(len(labels)))
    axes[2].set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    axes[2].set_yticklabels(labels, fontsize=8)
    axes[2].set_title("LR Confusion Matrix")
    for i in range(len(labels)):
        for j in range(len(labels)):
            axes[2].text(j, i, cm_lr[i, j], ha="center", va="center",
                         color="white" if cm_lr[i, j] > cm_lr.max() / 2 else "black", fontsize=8)

    plt.tight_layout()
    plt.savefig("model_evaluation.png", dpi=150, bbox_inches="tight")
    print("\n  📈 Evaluation plot saved: model_evaluation.png")

    # ── Sample predictions ─────────────────────────────────────────
    print("\n─" * 40)
    print("SAMPLE INTENT PREDICTIONS (Logistic Regression)")
    samples = [
        "What is the fee structure?",
        "Tell me about hostel facilities",
        "What are the placement statistics?",
        "Which courses are offered?",
        "hello there",
        "how can I contact the admission office?",
        "bye thanks",
        "random gibberish xyz abc",
    ]
    best_model = lr if lr_acc >= nb_acc else nb
    for s in samples:
        processed = preprocess(s)
        vec = vectorizer.transform([processed])
        intent = best_model.predict(vec)[0]
        conf   = best_model.predict_proba(vec).max()
        status = "✅" if conf >= 0.30 else "⚠️ LOW CONFIDENCE"
        print(f"  Input: '{s}' → Intent: {intent} ({conf*100:.1f}%) {status}")

    print("\n" + "=" * 60)
    print("  Training complete! Models ready for chatbot integration.")
    print("=" * 60)


if __name__ == "__main__":
    main()
