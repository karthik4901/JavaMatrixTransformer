package com.karthik.transformer.attention;

import com.karthik.transformer.core.Matrix;

/**
 * MultiHeadAttention — run attention multiple times in parallel, each looking
 * at the input from a different "perspective".
 *
 * Why multiple heads?
 *   Single attention head: ONE way of measuring token relationships.
 *   Multi-head attention: MANY simultaneous relationship patterns.
 *
 *   Head 1 might learn: subject-verb relationships ("paris is")
 *   Head 2 might learn: geographic associations ("paris" ↔ "france")
 *   Head 3 might learn: coreference            ("it" → "the country")
 *   Head 4 might learn: positional proximity    (nearby tokens)
 *
 * The model learns WHICH patterns are useful — we just give it the capacity.
 *
 * Architecture:
 *   dModel = total embedding dimension (e.g., 512)
 *   numHeads = number of attention heads (e.g., 8)
 *   dK = dModel / numHeads = per-head dimension (e.g., 64)
 *
 *   Each head operates in a smaller dK-dimensional space,
 *   then all heads are concatenated and projected back to dModel.
 *
 * Formula:
 *   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
 *   head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)
 *
 * @author Karthik Goud (Karthik Goud)
 */
public class MultiHeadAttention {

    private final int dModel;
    private final int numHeads;
    private final int dK;           // per-head key/query dim = dModel / numHeads
    private final int dV;           // per-head value dim = dModel / numHeads

    private final ScaledDotProductAttention[] heads;
    private final Matrix Wo;        // output projection: (numHeads*dV × dModel)

    public MultiHeadAttention(int dModel, int numHeads) {
        if (dModel % numHeads != 0)
            throw new IllegalArgumentException(
                "dModel (" + dModel + ") must be divisible by numHeads (" + numHeads + ")");

        this.dModel = dModel;
        this.numHeads = numHeads;
        this.dK = dModel / numHeads;
        this.dV = dModel / numHeads;

        // Each head has its own Q, K, V projection weights
        this.heads = new ScaledDotProductAttention[numHeads];
        for (int i = 0; i < numHeads; i++) {
            heads[i] = new ScaledDotProductAttention(dModel, dK, dV);
        }

        // Output projection — combines all heads back to dModel
        double scale = 1.0 / Math.sqrt(numHeads * dV);
        this.Wo = Matrix.random(numHeads * dV, dModel, scale);
    }

    /**
     * Forward pass — self-attention across all heads.
     *
     * @param input  Shape: (seqLen × dModel)
     * @return       Shape: (seqLen × dModel) — same shape in, same shape out
     */
    public Matrix forward(Matrix input) {
        int seqLen = input.rows;

        // Run each attention head independently
        // Each head output: (seqLen × dV)
        Matrix[] headOutputs = new Matrix[numHeads];
        for (int h = 0; h < numHeads; h++) {
            headOutputs[h] = heads[h].forward(input);
        }

        // Concatenate all head outputs along the feature dimension
        // Result shape: (seqLen × numHeads*dV) = (seqLen × dModel)
        Matrix concatenated = concatenateHeads(headOutputs, seqLen);

        // Final linear projection back to dModel
        // This allows the model to mix information across heads
        return concatenated.multiply(Wo);  // (seqLen × dModel)
    }

    /**
     * Masked multi-head attention — for autoregressive (decoder) models.
     * Each token can only attend to itself and previous tokens.
     */
    public Matrix maskedForward(Matrix input) {
        int seqLen = input.rows;
        Matrix[] headOutputs = new Matrix[numHeads];
        for (int h = 0; h < numHeads; h++) {
            headOutputs[h] = heads[h].maskedForward(input);
        }
        Matrix concatenated = concatenateHeads(headOutputs, seqLen);
        return concatenated.multiply(Wo);
    }

    /**
     * Concatenate multiple (seqLen × dV) matrices into (seqLen × numHeads*dV).
     */
    private Matrix concatenateHeads(Matrix[] headOutputs, int seqLen) {
        int totalDim = numHeads * dV;
        Matrix result = new Matrix(seqLen, totalDim);
        for (int h = 0; h < numHeads; h++) {
            for (int i = 0; i < seqLen; i++) {
                for (int j = 0; j < dV; j++) {
                    result.set(i, h * dV + j, headOutputs[h].get(i, j));
                }
            }
        }
        return result;
    }

    /**
     * Get attention weights for a specific head (for visualization).
     */
    public Matrix getHeadAttentionWeights(int headIndex) {
        return heads[headIndex].getLastAttentionWeights();
    }

    /**
     * Print attention patterns for a specific head.
     */
    public void printHeadAttention(int headIndex, String[] tokens) {
        System.out.printf("\n--- Head %d ---\n", headIndex + 1);
        heads[headIndex].printAttentionHeatmap(tokens);
    }

    /**
     * Count total trainable parameters in this multi-head attention block.
     */
    public int countParameters() {
        // Each head: 3 projection matrices of shape (dModel × dK)
        int perHeadParams = 3 * dModel * dK;
        int totalHeadParams = numHeads * perHeadParams;
        // Output projection: (numHeads*dV × dModel)
        int outputParams = numHeads * dV * dModel;
        return totalHeadParams + outputParams;
    }

    @Override
    public String toString() {
        return String.format(
            "MultiHeadAttention(dModel=%d, numHeads=%d, dK=%d, params=%d)",
            dModel, numHeads, dK, countParameters());
    }
}
