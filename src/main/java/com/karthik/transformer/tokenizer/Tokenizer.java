package com.karthik.transformer.tokenizer;

import java.util.List;

/**
 * Abstraction for text tokenization.
 * Implementations: {@link SimpleTokenizer} (word-level), future BPE, SentencePiece, etc.
 */
public interface Tokenizer {

    int[] encode(String text);

    String decode(int[] tokenIds);

    void buildVocabulary(List<String> corpus);

    int vocabularySize();

    int getId(String token);

    String getToken(int tokenId);

    boolean hasToken(String token);

    int eosId();
}
