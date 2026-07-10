package com.karthik.transformer.tokenizer;

import java.util.List;

/**
 * Text ↔ token ID contract.
 *
 * {@link SimpleTokenizer} is the current word-level implementation.
 * A BPE or SentencePiece implementation can plug in without changing
 * {@link com.karthik.transformer.model.TransformerModel}.
 */
public interface Tokenizer {

    /**
     * Encode text to token IDs (typically wrapped with BOS/EOS).
     *
     * @throws IllegalArgumentException if text is null (implementation-dependent)
     */
    int[] encode(String text);

    /** Decode IDs back to text, usually skipping special tokens. */
    String decode(int[] tokenIds);

    /** Grow vocabulary from a list of sentences (call once during setup). */
    void buildVocabulary(List<String> corpus);

    int vocabularySize();

    int getId(String token);

    String getToken(int tokenId);

    boolean hasToken(String token);

    /** End-of-sequence id used to stop generation. */
    int eosId();
}
