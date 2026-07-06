package com.karthik.transformer.attention;

import com.karthik.transformer.core.Matrix;

/**
 * TransformerBlock — one complete encoder layer of a transformer.
 *
 * A full GPT/BERT model stacks N of these blocks (GPT-3 has 96 of them).
 * Each block takes input, transforms it, and passes it to the next.
 * After many blocks, the model has deeply encoded the context of every token.
 *
 * Architecture of one block:
 *
 *   input
 *     │
 *     ├──────────────────────────┐
 *     │                          │  (residual connection)
 *     ▼                          │
 *   MultiHeadAttention           │
 *     │                          │
 *     └──────► Add & LayerNorm ◄─┘
 *                    │
 *     ┌──────────────┤
 *     │              │  (residual connection)
 *     ▼              │
 *   FeedForward      │
 *     │              │
 *     └──► Add & LayerNorm ◄────┘
 *                    │
 *                  output
 *
 * Residual connections (the "Add" part): output = layer(x) + x
 *   → Prevents vanishing gradients in deep networks
 *   → Lets information flow directly from input to output unchanged if needed
 *
 * FeedForward network: two linear layers with ReLU in between
 *   → dModel → dFF (typically 4× dModel) → dModel
 *   → Applied independently to each position
 *   → Adds non-linearity and capacity beyond attention
 *
 * @author Karthik Goud (Karthik Goud)
 */
public class TransformerBlock {

    private final MultiHeadAttention attention;
    private final Matrix W1;        // Feed-forward layer 1: (dModel × dFF)
    private final Matrix W2;        // Feed-forward layer 2: (dFF × dModel)
    private final int dModel;
    private final int dFF;
    private final double layerNormEpsilon = 1e-6;

    /**
     * @param dModel    Embedding dimension (e.g. 64)
     * @param numHeads  Attention heads (e.g. 4)
     * @param dFF       Feed-forward inner dimension (e.g. 256 = 4 × dModel)
     */
    public TransformerBlock(int dModel, int numHeads, int dFF) {
        this.dModel = dModel;
        this.dFF = dFF;
        this.attention = new MultiHeadAttention(dModel, numHeads);

        double scale = Math.sqrt(2.0 / (dModel + dFF));
        this.W1 = Matrix.random(dModel, dFF, scale);
        this.W2 = Matrix.random(dFF, dModel, scale);
    }

    /**
     * Forward pass through one transformer block.
     *
     * @param input  Shape: (seqLen × dModel)
     * @return       Shape: (seqLen × dModel) — same shape
     */
    public Matrix forward(Matrix input) {
        // ── Sub-layer 1: Multi-Head Self-Attention + Residual + LayerNorm ──
        Matrix attentionOutput = attention.forward(input);      // (seqLen × dModel)
        Matrix afterAttention = input.add(attentionOutput)      // residual: x + Attention(x)
                                     .layerNorm(layerNormEpsilon);

        // ── Sub-layer 2: Feed-Forward Network + Residual + LayerNorm ──
        Matrix ffOutput = feedForward(afterAttention);           // (seqLen × dModel)
        Matrix afterFF = afterAttention.add(ffOutput)           // residual: x + FF(x)
                                       .layerNorm(layerNormEpsilon);

        return afterFF;
    }

    /**
     * Masked variant for decoder (causal/autoregressive) transformers.
     * Prevents attending to future tokens.
     */
    public Matrix maskedForward(Matrix input) {
        Matrix attentionOutput = attention.maskedForward(input);
        Matrix afterAttention = input.add(attentionOutput)
                                     .layerNorm(layerNormEpsilon);
        Matrix ffOutput = feedForward(afterAttention);
        return afterAttention.add(ffOutput).layerNorm(layerNormEpsilon);
    }

    /**
     * Position-wise Feed-Forward Network.
     *
     * Applied to each token position independently (same weights, different inputs).
     * FF(x) = ReLU(x × W1) × W2
     *
     * The expand-then-contract shape (dModel → 4*dModel → dModel) gives the model
     * capacity to learn complex non-linear token transformations.
     */
    private Matrix feedForward(Matrix input) {
        // Expand: (seqLen × dModel) × (dModel × dFF) = (seqLen × dFF)
        Matrix expanded = input.multiply(W1).relu();
        // Contract: (seqLen × dFF) × (dFF × dModel) = (seqLen × dModel)
        return expanded.multiply(W2);
    }

    /**
     * Get the underlying attention block (for visualization).
     */
    public MultiHeadAttention getAttention() { return attention; }

    /**
     * Count all trainable parameters in this block.
     */
    public int countParameters() {
        int attentionParams = attention.countParameters();
        int ffParams = (dModel * dFF) + (dFF * dModel); // W1 + W2
        return attentionParams + ffParams;
    }

    @Override
    public String toString() {
        return String.format(
            "TransformerBlock(dModel=%d, dFF=%d, params=%d | %s)",
            dModel, dFF, countParameters(), attention);
    }
}
