import numpy as np
from numbers import Integral,Real
from scipy.sparse import vstack
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import GroupShuffleSplit

def create_text_groups(texts, threshold=0.95, batch_size=100):
    """Group near-duplicate texts using batched sparse similarity."""

    if (
        isinstance(threshold, (bool, np.bool_))
        or not isinstance(threshold, Real)
        or not np.isfinite(threshold)
        or not 0 < threshold <= 1
    ):
        raise ValueError(
            "threshold must be a finite number in (0, 1]."
        )

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or batch_size <= 0
    ):
        raise ValueError("batch_size must be a positive integer.")

    if isinstance(texts, (str, bytes)):
        raise ValueError("texts must be an iterable of non-empty strings.")
    try:
        texts = list(texts)
    except TypeError as error:
        raise ValueError("texts must be an iterable of non-empty strings.") from error
    if not all(isinstance(text, str) and bool(text.strip()) for text in texts):
        raise ValueError("Complaint narratives must be non-empty strings.")

    if not texts:
        return np.array([], dtype=int)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
    )

    vectors = vectorizer.fit_transform(texts)
    adjacency_batches = []

    for start in range(0, len(texts), batch_size):
        batch = vectors[start:start + batch_size]

        similarities = cosine_similarity(
            batch,
            vectors,
            dense_output=False,
        )

        similarities.data = (
            similarities.data >= threshold
        ).astype(np.int8)

        similarities.eliminate_zeros()
        adjacency_batches.append(similarities)

    adjacency = vstack(
        adjacency_batches,
        format="csr",
    )

    _, group_ids = connected_components(
        adjacency,
        directed=False,
    )

    return group_ids

def group_aware_split(X, y, groups, test_size=0.20, random_state=42):
    """Split data without sharing groups between train and test."""

    if len(X) != len(y) or len(X) != len(groups):
        raise ValueError("X, y, and groups must have equal lengths.")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")

    if len(np.unique(groups)) < 2:
        raise ValueError("At least two distinct groups are required.")

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=test_size,
        random_state=random_state,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    train_groups = set(np.asarray(groups)[train_idx])
    test_groups = set(np.asarray(groups)[test_idx])

    if train_groups & test_groups:
        raise RuntimeError("Group overlap detected.")

    return train_idx, test_idx
