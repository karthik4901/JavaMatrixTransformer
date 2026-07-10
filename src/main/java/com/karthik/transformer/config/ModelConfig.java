package com.karthik.transformer.config;

/**
 * Immutable model hyperparameters shared by demos, the Spring bean, and future training.
 *
 * {@link #DEMO} is intentionally tiny ({@code dModel=16}, one layer) so the
 * walkthrough runs quickly on a laptop. {@code dModel} must be divisible by
 * {@code numHeads} or construction fails.
 */
public record ModelConfig(
    int dModel,
    int numHeads,
    int dFF,
    int maxSeqLength,
    int numLayers
) {
    /** Small config used by demos and the geography API. */
    public static final ModelConfig DEMO = new ModelConfig(16, 4, 64, 64, 1);

    public ModelConfig {
        if (dModel <= 0 || numHeads <= 0 || dFF <= 0 || maxSeqLength <= 0 || numLayers <= 0) {
            throw new IllegalArgumentException("All model dimensions must be positive");
        }
        if (dModel % numHeads != 0) {
            throw new IllegalArgumentException(
                "dModel (" + dModel + ") must be divisible by numHeads (" + numHeads + ")");
        }
    }

    /** Per-head key/query dimension ({@code dModel / numHeads}). */
    public int headDim() {
        return dModel / numHeads;
    }
}
