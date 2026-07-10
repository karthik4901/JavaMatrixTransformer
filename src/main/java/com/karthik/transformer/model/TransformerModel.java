package com.karthik.transformer.model;

import com.karthik.transformer.attention.TransformerBlock;
import com.karthik.transformer.config.ModelConfig;
import com.karthik.transformer.core.Matrix;
import com.karthik.transformer.data.Corpus;
import com.karthik.transformer.embedding.EmbeddingLayer;
import com.karthik.transformer.encoding.PositionalEncoding;
import com.karthik.transformer.tokenizer.SimpleTokenizer;
import com.karthik.transformer.tokenizer.Tokenizer;

import java.util.ArrayList;
import java.util.List;

/**
 * Full forward pipeline: tokenize → embed → position → transformer layers → LM head.
 *
 * Built from a {@link Corpus} and {@link ModelConfig}. Weights are randomly
 * initialized; without training, {@link #predictNextToken(int[])} is not useful
 * for real answers. Geography Q&A currently uses corpus retrieval instead.
 */
public final class TransformerModel {

    private final ModelConfig config;
    private final Tokenizer tokenizer;
    private final EmbeddingLayer embedding;
    private final PositionalEncoding positionalEncoding;
    private final List<TransformerBlock> blocks;
    private final Matrix lmHead;

    private TransformerModel(
            ModelConfig config,
            Tokenizer tokenizer,
            EmbeddingLayer embedding,
            PositionalEncoding positionalEncoding,
            List<TransformerBlock> blocks,
            Matrix lmHead) {
        this.config = config;
        this.tokenizer = tokenizer;
        this.embedding = embedding;
        this.positionalEncoding = positionalEncoding;
        this.blocks = List.copyOf(blocks);
        this.lmHead = lmHead;
    }

    /**
     * Build vocabulary from the corpus, then allocate embeddings, PE, blocks, and LM head.
     */
    public static TransformerModel fromCorpus(Corpus corpus, ModelConfig config) {
        Tokenizer tokenizer = new SimpleTokenizer();
        tokenizer.buildVocabulary(corpus.sentences());

        EmbeddingLayer embedding = new EmbeddingLayer(tokenizer.vocabularySize(), config.dModel());
        PositionalEncoding positionalEncoding = new PositionalEncoding(
            config.maxSeqLength(), config.dModel());

        List<TransformerBlock> blocks = new ArrayList<>(config.numLayers());
        for (int i = 0; i < config.numLayers(); i++) {
            blocks.add(new TransformerBlock(config.dModel(), config.numHeads(), config.dFF()));
        }

        Matrix lmHead = Matrix.random(config.dModel(), tokenizer.vocabularySize(), 0.01);
        return new TransformerModel(config, tokenizer, embedding, positionalEncoding, blocks, lmHead);
    }

    /**
     * Contextual representations for each input token (shape: seqLen × dModel).
     *
     * @throws IllegalArgumentException if empty or longer than {@link ModelConfig#maxSeqLength()}
     */
    public Matrix forward(int[] tokenIds) {
        if (tokenIds == null || tokenIds.length == 0) {
            throw new IllegalArgumentException("tokenIds must not be empty");
        }
        if (tokenIds.length > config.maxSeqLength()) {
            throw new IllegalArgumentException(
                "Sequence length " + tokenIds.length + " exceeds max " + config.maxSeqLength());
        }

        Matrix hidden = positionalEncoding.addPositionalEncoding(embedding.embed(tokenIds));
        for (TransformerBlock block : blocks) {
            hidden = block.maskedForward(hidden);
        }
        return hidden;
    }

    /**
     * Greedy next-token id from the last position's context vector.
     * Meaningful only after training; used by demos of the generation path.
     */
    public int predictNextToken(int[] tokenIds) {
        Matrix contextual = forward(tokenIds);
        double[] lastContext = contextual.getRow(contextual.rows - 1);
        return argmax(projectLogits(lastContext));
    }

    private double[] projectLogits(double[] context) {
        int vocabSize = tokenizer.vocabularySize();
        double[] logits = new double[vocabSize];
        for (int j = 0; j < vocabSize; j++) {
            double sum = 0.0;
            for (int i = 0; i < config.dModel(); i++) {
                sum += context[i] * lmHead.get(i, j);
            }
            logits[j] = sum;
        }
        return logits;
    }

    private static int argmax(double[] values) {
        int best = 0;
        for (int i = 1; i < values.length; i++) {
            if (values[i] > values[best]) {
                best = i;
            }
        }
        return best;
    }

    /** Rough parameter count: embeddings + each block + LM head. */
    public int countParameters() {
        int total = embedding.getVocabularySize() * embedding.getEmbeddingDim();
        for (TransformerBlock block : blocks) {
            total += block.countParameters();
        }
        total += config.dModel() * tokenizer.vocabularySize();
        return total;
    }

    public ModelConfig config() { return config; }
    public Tokenizer tokenizer() { return tokenizer; }
    public EmbeddingLayer embedding() { return embedding; }
    public PositionalEncoding positionalEncoding() { return positionalEncoding; }
    public List<TransformerBlock> blocks() { return blocks; }

    @Override
    public String toString() {
        return String.format(
            "TransformerModel(layers=%d, vocab=%d, dModel=%d, params=%d)",
            blocks.size(), tokenizer.vocabularySize(), config.dModel(), countParameters());
    }
}
