package com.karthik.transformer.embedding;

import com.karthik.transformer.core.Matrix;

/**
 * EmbeddingLayer — converts discrete token IDs into dense vector representations.
 *
 * Think of it as a lookup table: each token ID maps to a row in a weight matrix.
 * The model LEARNS these vectors during training — words with similar meanings
 * end up with similar vectors (close in vector space).
 *
 * Example: in a geography corpus, "paris" and "france" may end up with
 * similar vectors after training; "paris" and "tokyo" would be farther apart.
 *
 * embeddingDim = how many attributes we track per token (typically 128–4096 in real LLMs)
 *
 * @author Karthik Goud (Karthik Goud)
 */
public class EmbeddingLayer {

    private final Matrix weights;  // shape: (vocabularySize × embeddingDim)
    private final int vocabularySize;
    private final int embeddingDim;

    /**
     * Initialize with random small values — like all neural network weights.
     * During training, backprop would adjust these. We initialize here for demo.
     */
    public EmbeddingLayer(int vocabularySize, int embeddingDim) {
        this.vocabularySize = vocabularySize;
        this.embeddingDim = embeddingDim;
        // Xavier initialization: scale = 1/sqrt(embeddingDim)
        this.weights = Matrix.random(vocabularySize, embeddingDim, 1.0 / Math.sqrt(embeddingDim));
    }

    /**
     * Lookup a single token ID — returns its embedding vector.
     * This is just reading row [tokenId] from the weight matrix.
     */
    public double[] lookup(int tokenId) {
        if (tokenId < 0 || tokenId >= vocabularySize)
            throw new IllegalArgumentException(
                "Token ID " + tokenId + " out of vocab range [0, " + vocabularySize + ")");
        return weights.getRow(tokenId);
    }

    /**
     * Embed a full sequence of token IDs.
     * Returns a matrix of shape (seqLen × embeddingDim).
     * This is the input to the transformer — one embedding row per token.
     */
    public Matrix embed(int[] tokenIds) {
        Matrix result = new Matrix(tokenIds.length, embeddingDim);
        for (int i = 0; i < tokenIds.length; i++) {
            double[] embedding = lookup(tokenIds[i]);
            for (int j = 0; j < embeddingDim; j++) {
                result.set(i, j, embedding[j]);
            }
        }
        return result;
    }

    /**
     * Cosine similarity between two token embeddings.
     * High value (~1.0) = semantically similar tokens.
     * Low value (~0.0) = unrelated tokens.
     * Negative (~-1.0) = opposite in meaning.
     */
    public double similarity(int tokenIdA, int tokenIdB) {
        return Matrix.cosineSimilarity(lookup(tokenIdA), lookup(tokenIdB));
    }

    /**
     * Find the N most similar tokens to a given token ID.
     * In a trained model, this would return semantically related words.
     */
    public int[] mostSimilar(int tokenId, int topN) {
        double[] query = lookup(tokenId);
        double[] scores = new double[vocabularySize];

        for (int i = 0; i < vocabularySize; i++) {
            if (i == tokenId) {
                scores[i] = -1.0; // exclude self
                continue;
            }
            scores[i] = Matrix.cosineSimilarity(query, weights.getRow(i));
        }

        // simple selection sort for top N
        int[] result = new int[topN];
        boolean[] used = new boolean[vocabularySize];
        for (int k = 0; k < topN; k++) {
            int best = -1;
            for (int i = 0; i < vocabularySize; i++) {
                if (!used[i] && (best == -1 || scores[i] > scores[best])) best = i;
            }
            result[k] = best;
            used[best] = true;
        }
        return result;
    }

    public int getVocabularySize() { return vocabularySize; }
    public int getEmbeddingDim() { return embeddingDim; }

    @Override
    public String toString() {
        return String.format("EmbeddingLayer(vocab=%d, dim=%d, params=%d)",
            vocabularySize, embeddingDim, vocabularySize * embeddingDim);
    }
}
