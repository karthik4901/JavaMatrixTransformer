package com.karthik.transformer.web;

import com.karthik.transformer.geography.GeographyAssistant;
import com.karthik.transformer.web.dto.AskRequest;
import com.karthik.transformer.web.dto.AskResponse;
import com.karthik.transformer.web.dto.HealthResponse;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.server.ResponseStatusException;

@RestController
public class GeographyController {

    private final GeographyAssistant assistant;

    public GeographyController(GeographyAssistant assistant) {
        this.assistant = assistant;
    }

    @GetMapping("/health")
    public HealthResponse health() {
        return new HealthResponse("ok", assistant.corpusSize(), assistant.modelSummary());
    }

    @GetMapping("/api/ask")
    public AskResponse askGet(@RequestParam(name = "q") String question) {
        return ask(question);
    }

    @PostMapping("/api/ask")
    public AskResponse askPost(@RequestBody AskRequest request) {
        if (request == null || request.question() == null || request.question().isBlank()) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, "Missing question field");
        }
        return ask(request.question());
    }

    private AskResponse ask(String question) {
        try {
            return new AskResponse(question, assistant.ask(question));
        } catch (IllegalArgumentException e) {
            throw new ResponseStatusException(HttpStatus.BAD_REQUEST, e.getMessage());
        }
    }
}
