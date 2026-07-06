package com.karthik.transformer.web;

import com.karthik.transformer.geography.GeographyAssistant;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.autoconfigure.web.servlet.WebMvcTest;
import org.springframework.http.MediaType;
import org.springframework.test.context.bean.override.mockito.MockitoBean;
import org.springframework.test.web.servlet.MockMvc;

import static org.mockito.Mockito.when;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get;
import static org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.jsonPath;
import static org.springframework.test.web.servlet.result.MockMvcResultMatchers.status;

@WebMvcTest(GeographyController.class)
class GeographyControllerTest {

    @Autowired
    private MockMvc mockMvc;

    @MockitoBean
    private GeographyAssistant geographyAssistant;

    @Test
    void healthReturnsOk() throws Exception {
        when(geographyAssistant.corpusSize()).thenReturn(103);
        when(geographyAssistant.modelSummary()).thenReturn("TransformerModel(test)");

        mockMvc.perform(get("/health"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.status").value("ok"))
            .andExpect(jsonPath("$.corpusSize").value(103));
    }

    @Test
    void askGetReturnsAnswer() throws Exception {
        when(geographyAssistant.ask("What is the capital of France?"))
            .thenReturn("Paris is the capital of France.");

        mockMvc.perform(get("/api/ask").param("q", "What is the capital of France?"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.answer").value("Paris is the capital of France."));
    }

    @Test
    void askPostReturnsAnswer() throws Exception {
        when(geographyAssistant.ask("What is the capital of Japan?"))
            .thenReturn("Tokyo is the capital of Japan.");

        mockMvc.perform(post("/api/ask")
                .contentType(MediaType.APPLICATION_JSON)
                .content("{\"question\":\"What is the capital of Japan?\"}"))
            .andExpect(status().isOk())
            .andExpect(jsonPath("$.answer").value("Tokyo is the capital of Japan."));
    }
}
