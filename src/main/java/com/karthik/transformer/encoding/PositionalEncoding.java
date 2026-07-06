package com.karthik.transformer.encoding;

import com.karthik.transformer.core.Matrix;

/**
 * PositionalEncoding — tells the transformer WHERE each token appears in the sequence.
 *
 * Problem: attention is order-agnostic. "Dog bites man" and "Man bites dog" would
 * produce identical embeddings without positional encoding — the model cannot
 * distinguish word order.
 *
 * Solution: add a unique positional signal to each token's embedding BEFORE
 * feeding into the transformer. We use sine and cosine waves at different
 * frequencies — the original "Attention Is All You Need" (Vaswani et al., 2017) approach.
 *
 * Formula:
 *   PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
 *   PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
 *
 * Where:
 *   pos      = position of the token in the sequence (0, 1, 2, ...)
 *   i        = dimension index (0, 1, 2, ... d_model/2)
 *   d_model  = embedding dimension
 *
 * Why sine/cosine?
 *   - Produces a unique pattern for every position
 *   - The model can learn to attend by relative position (sin/cos have this property)
 *   - Generalizes to sequences longer than seen during training
 *
 * @author Karthik Goud (Karthik Goud)
 */
public class PositionalEncoding {

    private final Matrix encodings; // shape: (maxSeqLen × embeddingDim)
    private final int maxSeqLen;
    private final int embeddingDim;

    public PositionalEncoding(int maxSeqLen, int embeddingDim) {
        this.maxSeqLen = maxSeqLen;
        this.embeddingDim = embeddingDim;
        this.encodings = compute(maxSeqLen, embeddingDim);
    }

    /**
     * Pre-compute the full positional encoding table.
     * In real transformers this is computed once and cached — never learned.
     */
    private static Matrix compute(int maxSeqLen, int dModel) {
        Matrix pe = new Matrix(maxSeqLen, dModel);
        for (int pos = 0; pos < maxSeqLen; pos++) {
            for (int i = 0; i < dModel / 2; i++) {
                // the "10000^(2i/dModel)" term controls the frequency
                double angle = pos / Math.pow(10000.0, (2.0 * i) / dModel);
                pe.set(pos, 2 * i,     Math.sin(angle)); // even dimensions = sin
                pe.set(pos, 2 * i + 1, Math.cos(angle)); // odd dimensions  = cos
            }
        }
        return pe;
    }

    /**
     * Add positional encodings to token embeddings.
     * Input shape:  (seqLen × embeddingDim)
     * Output shape: (seqLen × embeddingDim) — same shape, just shifted by position
     */
    public Matrix addPositionalEncoding(Matrix tokenEmbeddings) {
        if (tokenEmbeddings.rows > maxSeqLen) {
            throw new IllegalArgumentException(
                "Sequence length " + tokenEmbeddings.rows +
                " exceeds maximum " + maxSeqLen);
        }
        if (tokenEmbeddings.cols != embeddingDim) {
            throw new IllegalArgumentException(
                "Embedding dim mismatch: expected " + embeddingDim +
                ", got " + tokenEmbeddings.cols);
        }

        // Slice the precomputed table to match input sequence length
        Matrix posSlice = new Matrix(tokenEmbeddings.rows, embeddingDim);
        for (int i = 0; i < tokenEmbeddings.rows; i++)
            for (int j = 0; j < embeddingDim; j++)
                posSlice.set(i, j, encodings.get(i, j));

        // Element-wise add: token meaning + position information
        return tokenEmbeddings.add(posSlice);
    }

    /**
     * Get the raw encoding for a specific position (useful for inspection).
     */
    public double[] getEncoding(int position) {
        return encodings.getRow(position);
    }

    /**
     * Visualize how similar the positional encodings are between two positions.
     * Nearby positions should be more similar than distant positions.
     */
    public double positionSimilarity(int posA, int posB) {
        return Matrix.cosineSimilarity(getEncoding(posA), getEncoding(posB));
    }

    @Override
    public String toString() {
        return String.format("PositionalEncoding(maxSeqLen=%d, dim=%d)", maxSeqLen, embeddingDim);
    }
}
