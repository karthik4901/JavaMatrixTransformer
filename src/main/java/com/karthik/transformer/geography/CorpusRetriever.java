package com.karthik.transformer.geography;

import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.stream.Collectors;

/**
 * Finds the best-matching fact from a corpus for a natural-language question.
 * Uses keyword overlap — appropriate until the transformer is fully trained.
 */
final class CorpusRetriever {

    private static final Set<String> STOP_WORDS = Set.of(
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "what", "which", "who", "where", "when", "how", "why", "do", "does", "did",
        "in", "on", "at", "to", "for", "of", "and", "or", "tell", "me", "about"
    );

    private final List<String> sentences;

    CorpusRetriever(List<String> sentences) {
        this.sentences = List.copyOf(sentences);
    }

    String findBestAnswer(String question) {
        Set<String> keywords = extractKeywords(question);
        if (keywords.isEmpty()) {
            return fallbackMessage();
        }

        String bestSentence = null;
        int bestScore = 0;

        for (String sentence : sentences) {
            int score = scoreSentence(sentence, keywords, question);
            if (score > bestScore) {
                bestScore = score;
                bestSentence = sentence;
            }
        }

        if (bestSentence == null || bestScore < 10) {
            return fallbackMessage();
        }
        return bestSentence;
    }

    private int scoreSentence(String sentence, Set<String> keywords, String question) {
        Set<String> sentenceTokens = tokenize(sentence);
        int matches = 0;
        for (String keyword : keywords) {
            if (sentenceTokens.contains(keyword)) {
                matches++;
            }
        }
        if (matches == 0) {
            return 0;
        }

        int score = matches * 10;
        if (matches == keywords.size()) {
            score += 25;
        }

        String lowerQuestion = question.toLowerCase();
        String lowerSentence = sentence.toLowerCase();

        if (lowerQuestion.contains("capital") && lowerSentence.contains("capital")) {
            score += 20;
        }
        if (lowerQuestion.contains("longest") && lowerSentence.contains("longest")) {
            score += 20;
        }
        if (lowerQuestion.contains("largest") && lowerSentence.contains("largest")) {
            score += 20;
        }
        if (lowerQuestion.contains("highest") && lowerSentence.contains("highest")) {
            score += 20;
        }
        if (lowerQuestion.contains("continent") && lowerSentence.contains("continent")) {
            score += 15;
        }
        if (lowerQuestion.contains("ocean") && lowerSentence.contains("ocean")) {
            score += 15;
        }
        if (lowerQuestion.contains("river") && lowerSentence.contains("river")) {
            score += 15;
        }
        if (lowerQuestion.contains("mountain") && lowerSentence.contains("mountain")) {
            score += 15;
        }

        return score;
    }

    private static Set<String> extractKeywords(String question) {
        return tokenize(question).stream()
            .filter(token -> !STOP_WORDS.contains(token))
            .filter(token -> token.length() > 1)
            .collect(Collectors.toCollection(HashSet::new));
    }

    private static Set<String> tokenize(String text) {
        return Arrays.stream(text.toLowerCase()
                .replaceAll("[^a-z0-9\\s]", " ")
                .trim()
                .split("\\s+"))
            .filter(token -> !token.isEmpty())
            .collect(Collectors.toCollection(HashSet::new));
    }

    private static String fallbackMessage() {
        return "I don't have an answer for that in my geography knowledge base.";
    }
}
