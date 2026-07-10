package com.karthik.transformer.web.dto;

/**
 * Successful ask response.
 *
 * @param question echo of the client question
 * @param answer   best matching corpus fact (or fallback message)
 */
public record AskResponse(String question, String answer) {}
