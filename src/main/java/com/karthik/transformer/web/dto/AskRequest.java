package com.karthik.transformer.web.dto;

/**
 * POST /api/ask body.
 *
 * @param question natural-language geography question (required, non-blank)
 */
public record AskRequest(String question) {}
