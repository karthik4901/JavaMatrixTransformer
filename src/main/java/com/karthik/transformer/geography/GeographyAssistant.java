package com.karthik.transformer.geography;

import com.karthik.transformer.config.ModelConfig;
import com.karthik.transformer.data.GeographyCorpus;
import com.karthik.transformer.model.TransformerModel;

/**
 * Application-facing geography Q&A service.
 *
 * Builds a {@link TransformerModel} (for pipeline visibility and future training)
 * but answers questions through {@link CorpusRetriever} until weights are trained.
 * Shared by the Spring controller and the terminal chat.
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

    /** Load geography corpus, build model + retriever. */
    public static GeographyAssistant create() {
        GeographyCorpus corpus = new GeographyCorpus();
        TransformerModel model = TransformerModel.fromCorpus(corpus, ModelConfig.DEMO);
        return new GeographyAssistant(
            new CorpusRetriever(corpus.sentences()),
            model,
            corpus.sentences().size());
    }

    /**
     * Return the best matching fact for {@code question}.
     *
     * @throws IllegalArgumentException if the question is null or blank
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

    /** Short model description for health checks and logs. */
    public String modelSummary() {
        return model.toString();
    }
}
