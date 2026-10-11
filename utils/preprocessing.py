import re
import nltk

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Download required NLTK resources
try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords")

try:
    WordNetLemmatizer().lemmatize("test")
except LookupError:
    nltk.download("wordnet")


STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()


def clean_text(text):
    """
    Clean and preprocess resume/job-description text.

    Steps:
    1. Convert to lowercase
    2. Remove URLs
    3. Remove email addresses
    4. Remove special characters
    5. Remove extra spaces
    """

    if not text:
        return ""

    text = text.lower()

    # Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)

    # Keep letters, numbers and spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def tokenize_text(text):
    """Convert text into individual words."""

    cleaned_text = clean_text(text)

    tokens = cleaned_text.split()

    return tokens


def remove_stopwords(tokens):
    """Remove common English stopwords."""

    return [
        word
        for word in tokens
        if word not in STOP_WORDS
    ]


def lemmatize_tokens(tokens):
    """Convert words to their base form."""

    return [
        LEMMATIZER.lemmatize(word)
        for word in tokens
    ]


def preprocess_text(text):
    """
    Complete NLP preprocessing pipeline.

    Returns cleaned, normalized text.
    """

    tokens = tokenize_text(text)

    tokens = remove_stopwords(tokens)

    tokens = lemmatize_tokens(tokens)

    return " ".join(tokens)