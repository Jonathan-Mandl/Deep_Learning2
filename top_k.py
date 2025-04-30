import numpy as np

# Load vectors and vocabulary
vecs = np.loadtxt("embeddings/wordVectors.txt")
with open("embeddings/vocab.txt", "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f]

# Build word to index mapping
word2idx = {word: i for i, word in enumerate(vocab)}

# Normalize the vectors for efficient cosine computation
norms = np.linalg.norm(vecs, axis=1, keepdims=True)
normed_vecs = vecs / norms


def most_similar(word, k=5):
    word = word.lower()
    if word not in word2idx:
        print(f"'{word}' not found in vocabulary.")
        return []

    idx = word2idx[word]
    vec = normed_vecs[idx]
    sims = np.dot(normed_vecs, vec)

    # Exclude the word itself by masking or sorting
    sims[idx] = -1
    top_k = np.argsort(sims)[-k:][::-1]  # indices of top k
    return [(vocab[i], sims[i]) for i in top_k]


if __name__ == '__main__':
    query_words = ["dog", "england", "john", "explode", "office"]
    for w in query_words:
        print(f"\nTop 5 most similar to '{w}':")
        for i, (word, score) in enumerate(most_similar(w, k=5), 1):
            print(f"  {i}. {word} (cosine similarity: {score:.4f})")
