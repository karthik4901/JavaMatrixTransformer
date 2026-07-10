package com.karthik.transformer.attention;

import com.karthik.transformer.core.Matrix;

/**
 * One transformer layer: multi-head attention, residual + layer norm,
 * position-wise feed-forward, then residual + layer norm again.
 *
 * <pre>
 *   x → Attention → Add&Norm → FFN(dModel→dFF→dModel) → Add&Norm → y
 * </pre>
 *
 * Residuals keep gradients usable in deeper stacks. The FFN expands to
 * {@code dFF} (usually 4×dModel) then projects back, applied per position.
 * Stack many of these for BERT/GPT-scale depth; this project typically uses one.
 *
 * @author Karthik Goud
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
