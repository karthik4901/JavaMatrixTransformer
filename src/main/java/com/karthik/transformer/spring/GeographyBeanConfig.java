package com.karthik.transformer.spring;

import com.karthik.transformer.geography.GeographyAssistant;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

@Configuration
public class GeographyBeanConfig {

    @Bean
    GeographyAssistant geographyAssistant() {
        return GeographyAssistant.create();
    }
}
