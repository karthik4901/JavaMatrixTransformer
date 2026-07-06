package com.karthik.transformer.demo;

import com.karthik.transformer.geography.GeographyChat;

/**
 * Entry point for non-Spring demos: internals walkthrough, batch Q&A, terminal chat.
 * For the REST API, use {@link com.karthik.transformer.JavaMatrixTransformerApplication}.
 */
public final class Main {

    public static void main(String[] args) {
        if (args.length == 0) {
            Demo.main(args);
            return;
        }

        switch (args[0].toLowerCase()) {
            case "geography" -> GeographyQADemo.main(args);
            case "chat"      -> GeographyChat.main(args);
            default          -> Demo.main(args);
        }
    }
}
