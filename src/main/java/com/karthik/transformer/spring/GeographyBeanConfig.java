package com.karthik.transformer.spring;

import com.karthik.transformer.geography.GeographyAssistant;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * Wires a single shared {@link GeographyAssistant} for the web layer.
 *
 * Created once at startup so corpus load and model init are not repeated per request.
 */
@Configuration
public class GeographyBeanConfig {

    @Bean
    GeographyAssistant geographyAssistant() {
        return GeographyAssistant.create();
    }
}
