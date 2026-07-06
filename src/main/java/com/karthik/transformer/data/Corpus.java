package com.karthik.transformer.data;

import java.util.List;

/**
 * Training or demo corpus — pluggable data source for any domain.
 */
public interface Corpus {

    String name();

    String description();

    List<String> sentences();

    List<String> sampleQuestions();
}
