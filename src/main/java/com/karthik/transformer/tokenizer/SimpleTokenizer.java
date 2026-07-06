package com.karthik.transformer.tokenizer;

import java.util.*;

/**
 * SimpleTokenizer — converts raw text into token IDs and back.
 *
 * Real LLMs use Byte Pair Encoding (BPE) or WordPiece. This implementation
 * uses word-level tokenization to make the concept crystal clear.
 * The principle is identical — map text units to integer IDs so the model
 * can work with numbers instead of strings.
 *
 * Special tokens:
 *   [PAD] = 0  — padding to make all sequences the same length
 *   [UNK] = 1  — unknown token (word not in vocabulary)
 *   [BOS] = 2  — beginning of sequence
 *   [EOS] = 3  — end of sequence
 *
 * @author Karthik Goud (Karthik Goud)
 */
public class SimpleTokenizer implements Tokenizer {

    public static final String PAD = "[PAD]";
    public static final String UNK = "[UNK]";
    public static final String BOS = "[BOS]";
    public static final String EOS = "[EOS]";

    public static final int PAD_ID = 0;
    public static final int UNK_ID = 1;
    public static final int BOS_ID = 2;
    public static final int EOS_ID = 3;

    private final Map<String, Integer> wordToId = new LinkedHashMap<>();
    private final Map<Integer, String> idToWord = new LinkedHashMap<>();
    private int nextId = 4; // 0-3 reserved for special tokens

    public SimpleTokenizer() {
        // register special tokens
        register(PAD, PAD_ID);
        register(UNK, UNK_ID);
        register(BOS, BOS_ID);
        register(EOS, EOS_ID);
    }

    /**
     * Build vocabulary from a corpus of text.
     * Splits on whitespace and punctuation, lowercases everything.
     */
    @Override
    public void buildVocabulary(List<String> corpus) {
        for (String sentence : corpus) {
            for (String token : tokenize(sentence)) {
                if (!wordToId.containsKey(token)) {
                    register(token, nextId++);
                }
            }
        }
    }

    /**
     * Encode a sentence to token IDs.
     * Wraps with [BOS] and [EOS] by default.
     */
    @Override
    public int[] encode(String sentence) {
        String[] words = tokenize(sentence);
        int[] ids = new int[words.length + 2]; // +2 for BOS, EOS
        ids[0] = BOS_ID;
        for (int i = 0; i < words.length; i++) {
            ids[i + 1] = wordToId.getOrDefault(words[i], UNK_ID);
        }
        ids[ids.length - 1] = EOS_ID;
        return ids;
    }

    /**
     * Encode without BOS/EOS wrapping.
     */
    public int[] encodeRaw(String sentence) {
        String[] words = tokenize(sentence);
        int[] ids = new int[words.length];
        for (int i = 0; i < words.length; i++) {
            ids[i] = wordToId.getOrDefault(words[i], UNK_ID);
        }
        return ids;
    }

    /**
     * Decode token IDs back to a sentence.
     * Skips special tokens.
     */
    @Override
    public String decode(int[] ids) {
        StringBuilder sb = new StringBuilder();
        for (int id : ids) {
            String word = idToWord.getOrDefault(id, UNK);
            if (!word.equals(PAD) && !word.equals(BOS) && !word.equals(EOS)) {
                if (sb.length() > 0) sb.append(" ");
                sb.append(word);
            }
        }
        return sb.toString();
    }

    /**
     * Pad or truncate a sequence to a fixed length.
     * Real transformers need fixed-length input for batching.
     */
    public int[] padOrTruncate(int[] ids, int maxLength) {
        if (ids.length == maxLength) return ids;
        int[] result = new int[maxLength];
        int copyLen = Math.min(ids.length, maxLength);
        System.arraycopy(ids, 0, result, 0, copyLen);
        // remaining slots stay 0 = PAD_ID
        return result;
    }

    @Override
    public int vocabularySize() { return nextId; }

    @Override
    public int eosId() { return EOS_ID; }
    @Override
    public boolean hasToken(String token) { return wordToId.containsKey(token); }

    @Override
    public int getId(String token) { return wordToId.getOrDefault(token, UNK_ID); }

    @Override
    public String getToken(int id) { return idToWord.getOrDefault(id, UNK); }

    /** Split sentence into lowercase word tokens */
    private String[] tokenize(String sentence) {
        return sentence.toLowerCase()
            .replaceAll("[^a-z0-9\\s]", " ")
            .trim()
            .split("\\s+");
    }

    private void register(String word, int id) {
        wordToId.put(word, id);
        idToWord.put(id, word);
    }

    @Override
    public String toString() {
        return String.format("SimpleTokenizer(vocab_size=%d)", nextId);
    }
}
