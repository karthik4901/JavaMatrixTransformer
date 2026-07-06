package com.karthik.transformer.demo;

import com.karthik.transformer.data.GeographyCorpus;
import com.karthik.transformer.geography.GeographyAssistant;

/**
 * Batch geography Q&A — runs sample questions from the corpus.
 *
 * Run: {@code gradle geography}
 */
public final class GeographyQADemo {

    public static void main(String[] args) {
        GeographyCorpus corpus = new GeographyCorpus();

        printBanner();
        printStep("LOAD TRAINING DATA");
        System.out.printf("%n✓ Loaded %d geography sentences%n", corpus.sentences().size());
        System.out.println("Sample:");
        for (int i = 0; i < 3; i++) {
            System.out.printf("  [%d] %s%n", i + 1, corpus.sentences().get(i));
        }

        printStep("BUILD TRANSFORMER MODEL");
        GeographyAssistant assistant = GeographyAssistant.create();
        System.out.println(assistant.modelSummary());

        printStep("ANSWER GEOGRAPHY QUESTIONS");
        for (String question : corpus.sampleQuestions()) {
            System.out.printf("%n❓ %s%n", question);
            System.out.printf("→ %s%n", assistant.ask(question));
        }

        printStep("TRY INTERACTIVE MODES");
        System.out.println("\n  gradle bootRun   # Spring Boot REST API");
        System.out.println("  gradle chat      # interactive terminal");
        System.out.println("\n" + "═".repeat(60));
    }

    private static void printBanner() {
        System.out.println("\n╔══════════════════════════════════════════════════════════╗");
        System.out.println("║   World Geography QA — JavaMatrixTransformer             ║");
        System.out.println("║   Learn transformers like you learned Java: step by step ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝\n");
    }

    private static void printStep(String title) {
        System.out.println("═".repeat(60));
        System.out.println("  " + title);
        System.out.println("═".repeat(60));
    }
}
