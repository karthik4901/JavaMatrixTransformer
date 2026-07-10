package com.karthik.transformer;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Spring Boot entry point for the geography REST API.
 *
 * Run with {@code ./gradlew bootRun}. Default port is 8080
 * ({@code src/main/resources/application.properties}).
 */
@SpringBootApplication
public class JavaMatrixTransformerApplication {

    public static void main(String[] args) {
        SpringApplication.run(JavaMatrixTransformerApplication.class, args);
    }
}
