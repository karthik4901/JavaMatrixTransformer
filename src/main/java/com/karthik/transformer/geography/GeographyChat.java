package com.karthik.transformer.geography;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;

/**
 * Interactive terminal REPL for geography questions.
 *
 * Run: {@code ./gradlew chat}
 * Type {@code exit}, {@code quit}, or {@code q} to stop.
 */
public final class GeographyChat {

    public static void main(String[] args) {
        System.out.println();
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║   Geography Chat — JavaMatrixTransformer                 ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
        System.out.println();
        System.out.println("Loading geography model...");

        GeographyAssistant assistant = GeographyAssistant.create();
        System.out.printf("Ready — %d facts loaded%n", assistant.corpusSize());
        System.out.println(assistant.modelSummary());
        System.out.println();
        System.out.println("Type a geography question, or 'exit' to quit.");
        System.out.println();

        try (BufferedReader reader = new BufferedReader(
                new InputStreamReader(System.in, StandardCharsets.UTF_8))) {

            while (true) {
                System.out.println("────────────────────────────────────────");
                System.out.print("Question: ");

                String question = reader.readLine();
                if (question == null) {
                    System.out.println();
                    System.out.println("Session ended.");
                    break;
                }

                question = question.trim();
                if (question.isEmpty()) {
                    continue;
                }
                if (isExitCommand(question)) {
                    System.out.println("Goodbye!");
                    break;
                }

                try {
                    String answer = assistant.ask(question);
                    System.out.println("Answer:  " + answer);
                } catch (IllegalArgumentException e) {
                    System.out.println("Answer:  Please enter a valid question.");
                }
            }
        } catch (Exception e) {
            System.out.println();
            System.out.println("Could not read input. Try:");
            System.out.println("  gradle chat");
            System.out.println("  ./build/scripts/JavaMatrixTransformer chat");
        }
    }

    private static boolean isExitCommand(String input) {
        String lower = input.toLowerCase();
        return lower.equals("exit") || lower.equals("quit") || lower.equals("q");
    }
}
