"""Small document-classification helpers used by the Streamlit app."""

from __future__ import annotations

import csv
from io import BytesIO
from pathlib import Path
from typing import BinaryIO

from docx import Document
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

DATA_PATH = Path(__file__).with_name("sample_documents.csv")
MAX_DOCUMENT_CHARACTERS = 100_000


def load_training_examples(path: Path = DATA_PATH) -> tuple[list[str], list[str]]:
    """Read example document text and labels from the CSV training file."""
    texts: list[str] = []
    labels: list[str] = []

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        for row in csv.DictReader(csv_file):
            text = row["text"].strip()
            label = row["label"].strip()
            if text and label:
                texts.append(text)
                labels.append(label)

    if len(set(labels)) < 2:
        raise ValueError("The training file must contain at least two document types.")

    return texts, labels


def train_classifier(texts: list[str], labels: list[str]) -> Pipeline:
    """Teach a text model to predict a label from example documents."""
    if len(texts) != len(labels) or not texts:
        raise ValueError("Training needs the same non-zero number of texts and labels.")
    if len(set(labels)) < 2:
        raise ValueError("Training needs examples from at least two document types.")

    model = Pipeline(
        [
            ("words", TfidfVectorizer(ngram_range=(1, 2), stop_words="english")),
            ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    model.fit(texts, labels)
    return model


def classify_text(model: Pipeline, text: str) -> tuple[str, float, dict[str, float]]:
    """Return the most likely document type and each type's model score."""
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Please add some document text before classifying it.")

    probabilities = model.predict_proba([cleaned_text])[0]
    labels = model.named_steps["classifier"].classes_
    scores = {
        str(label): float(probability)
        for label, probability in zip(labels, probabilities)
    }
    predicted_label = max(scores, key=scores.get)
    return predicted_label, scores[predicted_label], scores


def extract_document_text(filename: str, file_data: bytes | BinaryIO) -> str:
    """Extract text from a TXT, PDF, or DOCX file."""
    extension = Path(filename).suffix.lower()
    data = file_data if isinstance(file_data, bytes) else file_data.read()

    if extension == ".txt":
        try:
            text = data.decode("utf-8-sig")
        except UnicodeDecodeError as error:
            raise ValueError("This text file is not saved as UTF-8.") from error
    elif extension == ".pdf":
        try:
            pdf = PdfReader(BytesIO(data))
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
        except Exception as error:
            raise ValueError("I could not read this PDF. Please try another file.") from error
    elif extension == ".docx":
        try:
            document = Document(BytesIO(data))
            text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        except Exception as error:
            raise ValueError("I could not read this Word file. Please try another file.") from error
    else:
        raise ValueError("Please choose a .txt, .pdf, or .docx file.")

    text = text.strip()
    if not text:
        raise ValueError("No readable text was found in that file.")
    if len(text) > MAX_DOCUMENT_CHARACTERS:
        raise ValueError("This document is too long. Please use a file under 100,000 characters.")
    return text
