package com.karthik.transformer.config;

/**
 * Immutable hyperparameters for a language model.
 * Centralizes configuration so demos, tests, and future training share one source of truth.
 */
public record ModelConfig(
    int dModel,
    int numHeads,
    int dFF,
    int maxSeqLength,
    int numLayers
) {
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

    public int headDim() {
        return dModel / numHeads;
    }
}
