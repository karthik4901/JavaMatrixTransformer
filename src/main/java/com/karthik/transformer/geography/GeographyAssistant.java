package com.karthik.transformer.geography;

import com.karthik.transformer.config.ModelConfig;
import com.karthik.transformer.data.GeographyCorpus;
import com.karthik.transformer.model.TransformerModel;

/**
 * Answers geography questions by retrieving the best-matching fact from the corpus.
 * The transformer model is loaded for the pipeline demo; answers come from the
 * knowledge base until full training is implemented.
 */
public final class GeographyAssistant {

    private final CorpusRetriever retriever;
    private final TransformerModel model;
    private final int sentenceCount;

    private GeographyAssistant(CorpusRetriever retriever, TransformerModel model, int sentenceCount) {
        this.retriever = retriever;
        this.model = model;
        this.sentenceCount = sentenceCount;
    }

    public static GeographyAssistant create() {
        GeographyCorpus corpus = new GeographyCorpus();
        TransformerModel model = TransformerModel.fromCorpus(corpus, ModelConfig.DEMO);
        return new GeographyAssistant(
            new CorpusRetriever(corpus.sentences()),
            model,
            corpus.sentences().size());
    }

    /**
     * Answer a geography question using corpus retrieval.
     */
    public String ask(String question) {
        if (question == null || question.isBlank()) {
            throw new IllegalArgumentException("Question must not be blank");
        }
        return retriever.findBestAnswer(question.trim());
    }

    public int corpusSize() {
        return sentenceCount;
    }

    public String modelSummary() {
        return model.toString();
    }
}
