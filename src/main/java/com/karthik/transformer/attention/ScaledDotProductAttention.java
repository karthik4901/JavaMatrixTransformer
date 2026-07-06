package com.karthik.transformer.attention;

import com.karthik.transformer.core.Matrix;

/**
 * ScaledDotProductAttention — the heart of every transformer model.
 *
 * "Attention Is All You Need" (Vaswani et al., 2017) — this single mechanism
 * replaced recurrent networks (LSTMs, GRUs) and became the foundation of
 * GPT, BERT, Claude, and every modern LLM.
 *
 * The core idea: for each token, figure out which other tokens are most
 * relevant, then create a weighted blend of their information.
 *
 * Three matrices are involved:
 *   Q (Query)  — "What am I looking for?"
 *   K (Key)    — "What do I have to offer?"
 *   V (Value)  — "What information do I actually contain?"
 *
 * Example: token "capital" attending to "paris" and "france" in
 * "paris is the capital of france" — Q asks what relates, K advertises identity,
 * V carries meaning, high-scoring pairs get blended into the output.
 *
 * The full formula:
 *   Attention(Q, K, V) = softmax(Q × K^T / sqrt(d_k)) × V
 *
 *   Step 1: Q × K^T        = raw similarity scores (which tokens relate to which)
 *   Step 2: / sqrt(d_k)    = scale down to prevent softmax saturation
 *   Step 3: softmax(...)   = convert scores to attention weights (sum to 1)
 *   Step 4: weights × V    = weighted blend of value vectors
 *
 * @author Karthik Goud (Karthik Goud)
 */
public class ScaledDotProductAttention {

    private final int dModel;   // embedding dimension
    private final int dK;       // key/query dimension (often dModel / numHeads)
    private final int dV;       // value dimension

    // Learned projection weights (in real training, these are updated via backprop)
    private final Matrix Wq;    // Query projection:  (dModel × dK)
    private final Matrix Wk;    // Key projection:    (dModel × dK)
    private final Matrix Wv;    // Value projection:  (dModel × dV)

    // Store last attention weights for inspection/visualization
    private Matrix lastAttentionWeights;

    public ScaledDotProductAttention(int dModel, int dK, int dV) {
        this.dModel = dModel;
        this.dK = dK;
        this.dV = dV;

        double scale = 1.0 / Math.sqrt(dK);
        this.Wq = Matrix.random(dModel, dK, scale);
        this.Wk = Matrix.random(dModel, dK, scale);
        this.Wv = Matrix.random(dModel, dV, scale);
    }

    /**
     * Forward pass: compute attention output.
     *
     * @param input  Token embeddings + positional encoding — shape: (seqLen × dModel)
     * @return       Attended output — shape: (seqLen × dV)
     *
     * Each output row is a context-aware blend of the input,
     * weighted by how much attention each token pays to every other token.
     */
    public Matrix forward(Matrix input) {
        return forward(input, input, input); // self-attention: Q=K=V=input
    }

    /**
     * Cross-attention variant: queries come from one source, keys/values from another.
     * Used in encoder-decoder transformers (e.g., translation models).
     *
     * @param queryInput  Source for Q — shape: (seqLenQ × dModel)
     * @param keyInput    Source for K — shape: (seqLenK × dModel)
     * @param valueInput  Source for V — shape: (seqLenK × dModel)
     */
    public Matrix forward(Matrix queryInput, Matrix keyInput, Matrix valueInput) {
        // Step 1: Project inputs into Q, K, V spaces
        // These projections allow the model to learn different "views" of the input
        Matrix Q = queryInput.multiply(Wq);  // (seqLen × dK)
        Matrix K = keyInput.multiply(Wk);    // (seqLen × dK)
        Matrix V = valueInput.multiply(Wv);  // (seqLen × dV)

        // Step 2: Compute raw attention scores — Q × K^T
        // Result shape: (seqLenQ × seqLenK)
        // Each cell [i][j] = "how much does token i want to attend to token j?"
        Matrix scores = Q.multiply(K.transpose());  // (seqLen × seqLen)

        // Step 3: Scale by 1/sqrt(dK) to prevent softmax from saturating
        // Without this, for large dK, dot products grow large → softmax → near one-hot → no learning
        double scaleFactor = 1.0 / Math.sqrt(dK);
        Matrix scaledScores = scores.scale(scaleFactor);

        // Step 4: Apply softmax row-wise — each row becomes a probability distribution
        // Row i tells us: "token i attends to token j with weight w[i][j]"
        // All weights in a row sum to 1.0
        Matrix attentionWeights = scaledScores.softmax();
        this.lastAttentionWeights = attentionWeights; // save for visualization

        // Step 5: Weighted sum of Value vectors
        // Output[i] = sum over j of (attentionWeight[i][j] * V[j])
        // Each token's output is a blend of ALL tokens' values, weighted by relevance
        return attentionWeights.multiply(V);  // (seqLen × dV)
    }

    /**
     * Masked attention — prevents tokens from attending to future positions.
     * This is what makes GPT-style models autoregressive (can only see the past).
     *
     * The mask sets future positions to -∞ before softmax,
     * so their attention weights become ~0 after softmax.
     */
    public Matrix maskedForward(Matrix input) {
        Matrix Q = input.multiply(Wq);
        Matrix K = input.multiply(Wk);
        Matrix V = input.multiply(Wv);

        Matrix scores = Q.multiply(K.transpose()).scale(1.0 / Math.sqrt(dK));

        // Apply causal mask: set upper triangle to -infinity
        Matrix masked = applyMask(scores);
        Matrix attentionWeights = masked.softmax();
        this.lastAttentionWeights = attentionWeights;

        return attentionWeights.multiply(V);
    }

    /**
     * Sets the upper-right triangle of the score matrix to -1e9 (effectively -∞).
     * After softmax, these positions get weight ≈ 0.
     */
    private Matrix applyMask(Matrix scores) {
        Matrix masked = new Matrix(scores.rows, scores.cols);
        for (int i = 0; i < scores.rows; i++)
            for (int j = 0; j < scores.cols; j++)
                masked.set(i, j, j > i ? -1e9 : scores.get(i, j));
        return masked;
    }

    /**
     * Get the attention weight matrix from the last forward pass.
     * Useful for attention visualization — see which tokens attend to which.
     */
    public Matrix getLastAttentionWeights() {
        return lastAttentionWeights;
    }

    /**
     * Print a readable attention heatmap for a sequence of tokens.
     * Higher values = more attention paid.
     */
    public void printAttentionHeatmap(String[] tokens) {
        if (lastAttentionWeights == null) {
            System.out.println("No attention computed yet. Run forward() first.");
            return;
        }
        int n = Math.min(tokens.length, lastAttentionWeights.rows);
        System.out.println("\n=== Attention Heatmap ===");
        System.out.printf("%-12s", "");
        for (int j = 0; j < n; j++) System.out.printf("%-10s", tokens[j]);
        System.out.println();
        for (int i = 0; i < n; i++) {
            System.out.printf("%-12s", tokens[i]);
            for (int j = 0; j < n; j++) {
                double w = lastAttentionWeights.get(i, j);
                // Visual bar: ████ for high attention, ░░░░ for low
                String bar = w > 0.5 ? "████" : w > 0.3 ? "███░" : w > 0.15 ? "██░░" : w > 0.05 ? "█░░░" : "░░░░";
                System.out.printf("%-10s", bar);
            }
            System.out.println();
        }
    }

    @Override
    public String toString() {
        return String.format("ScaledDotProductAttention(dModel=%d, dK=%d, dV=%d, params=%d)",
            dModel, dK, dV, (dModel * dK * 3) + (dV > dK ? dModel * dV : 0));
    }
}
