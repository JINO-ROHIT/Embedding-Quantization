import json
import os
import time

import faiss
import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sentence_transformers.quantization import quantize_embeddings
from usearch.index import Index


model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1",
    prompts={"query": "Represent this sentence for searching relevant passages: "},
    default_prompt_name="query",
)

dataset = load_dataset("quora", split="train").map(
    lambda batch: {"text": [text for sample in batch["questions"] for text in sample["text"]]},
    batched=True,
    remove_columns=["questions", "is_duplicate"],
)
max_corpus_size = 100_000
corpus = dataset["text"][:max_corpus_size]

query = "How do I become a good programmer?"

if os.path.exists("quora_faiss_ubinary.index"):
    binary_index = faiss.read_index_binary("quora_faiss_ubinary.index")
    index_normal = faiss.read_index("quora_faiss_normal.index")
    int8_view = Index.restore("quora_usearch_int8.index", view=True)

else:
    full_corpus_embeddings = model.encode(corpus, normalize_embeddings=True, show_progress_bar=True)

    index_normal = faiss.IndexFlatL2(1024)
    index_normal.add(full_corpus_embeddings)
    faiss.write_index(index_normal, "quora_faiss_normal.index")

    ubinary_embeddings = quantize_embeddings(full_corpus_embeddings, "ubinary")
    binary_index = faiss.IndexBinaryFlat(1024)
    binary_index.add(ubinary_embeddings)
    faiss.write_index_binary(binary_index, "quora_faiss_ubinary.index")

    int8_embeddings = quantize_embeddings(full_corpus_embeddings, "int8")
    index = Index(ndim=1024, dtype="i8")
    index.add(np.arange(len(int8_embeddings)), int8_embeddings)
    index.save("quora_usearch_int8.index")
    del index

    int8_view = Index.restore("quora_usearch_int8.index", view=True)


def search(query, top_k: int = 20, rescore_multiplier: int = 10):


    # Embed the query as float32
    start_time = time.time()
    query_embedding = model.encode(query)
    embed_time = time.time() - start_time

    # Quantize the query to ubinary
    start_time = time.time()
    query_embedding_ubinary = quantize_embeddings(query_embedding.reshape(1, -1), "ubinary")
    quantize_time = time.time() - start_time

    # Search the binary index
    start_time = time.time()
    _scores, binary_ids = binary_index.search(query_embedding_ubinary, top_k * rescore_multiplier)
    binary_ids = binary_ids[0]
    search_time = time.time() - start_time

    # search the normal index
    start_time = time.time()
    _, _ = index_normal.search(query_embedding.reshape(1, -1), top_k * rescore_multiplier)
    normal_search_time = time.time() - start_time


    # Load the corresponding int8 embeddings
    start_time = time.time()
    int8_embeddings = int8_view[binary_ids].astype(int)
    load_time = time.time() - start_time

    # Rescore the top_k * rescore_multiplier using the float32 query embedding and the int8 document embeddings
    start_time = time.time()
    scores = query_embedding @ int8_embeddings.T
    rescore_time = time.time() - start_time

    # Sort the scores and return the top_k
    start_time = time.time()
    indices = (-scores).argsort()[:top_k]
    top_k_indices = binary_ids[indices]
    top_k_scores = scores[indices]
    sort_time = time.time() - start_time

    return (
        top_k_scores.tolist(),
        top_k_indices.tolist(),
        {
            "Embed Time": f"{embed_time:.4f} s",
            "Quantize Time": f"{quantize_time:.4f} s",
            "Search Time": f"{search_time:.4f} s",
            "float32 Search Time" : f"{normal_search_time:.4f} s",
            "Load Time": f"{load_time:.4f} s",
            "Rescore Time": f"{rescore_time:.4f} s",
            "Sort Time": f"{sort_time:.4f} s",
            "Total Retrieval Time": f"{quantize_time + search_time + load_time + rescore_time + sort_time:.4f} s",
        },
    )


while True:
    scores, indices, timings = search(query)

    print(f"Timings:\n{json.dumps(timings, indent=2)}")
    print(f"Query: {query}")
    for score, index in zip(scores, indices):
        print(f"(Score: {score:.4f}) {corpus[index]}")
    print("")

    query = input("Please enter a question: ")
