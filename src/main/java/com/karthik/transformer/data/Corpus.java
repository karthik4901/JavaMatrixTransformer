package com.karthik.transformer.data;

import java.util.List;

/**
 * Domain text source for vocabulary building and Q&A demos.
 *
 * Swap {@link GeographyCorpus} for another domain without changing the model
 * or Spring controller — only the bean / factory that supplies the corpus.
 */
public interface Corpus {

    String name();

    String description();

    /** Fact sentences used to build vocabulary and (today) answer retrieval. */
    List<String> sentences();

    /** Example questions for batch demos and docs. */
    List<String> sampleQuestions();
}
