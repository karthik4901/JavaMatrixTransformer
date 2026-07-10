package com.karthik.transformer.web.dto;

/**
 * Liveness payload for GET /health.
 *
 * @param status     always {@code "ok"} when the process is up
 * @param corpusSize number of geography facts loaded
 * @param model      short {@link com.karthik.transformer.model.TransformerModel} summary
 */
public record HealthResponse(String status, int corpusSize, String model) {}
