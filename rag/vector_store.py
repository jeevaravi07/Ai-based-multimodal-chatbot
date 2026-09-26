"""
Step 3 of the case-report RAG pipeline: FAISS indices over the case-text
embeddings and the case-image embeddings, plus the metadata needed to go
from "nearest vector" back to a real case row.
"""
import pickle

import faiss
import numpy as np

import config


def build_index(embeddings: np.ndarray) -> faiss.Index:
    embeddings = np.ascontiguousarray(embeddings.astype(np.float32))
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index

def save_index(index: faiss.Index, path: str):
    faiss.write_index(index, path)


def load_index(path: str) -> faiss.Index:
    return faiss.read_index(path)


def save_metadata(records: list, path: str = config.METADATA_PKL):
    with open(path, "wb") as f:
        pickle.dump(records, f)


def load_metadata(path: str = config.METADATA_PKL) -> list:
    with open(path, "rb") as f:
        return pickle.load(f)


def search(index: faiss.Index, query_vec: np.ndarray, top_k: int = config.TOP_K_CASES):
    query_vec = np.ascontiguousarray(query_vec.astype(np.float32)).reshape(1, -1)
    distances, indices = index.search(query_vec, top_k)
    return distances[0], indices[0]
