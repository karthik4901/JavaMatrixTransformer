package com.karthik.transformer.data;

import java.util.List;

/**
 * Fixed set of world-geography facts used for vocabulary and retrieval-based Q&A.
 *
 * Chosen as a first domain because answers are short and easy to verify.
 * Not a trained knowledge base — just declarative sentences scored by
 * {@link com.karthik.transformer.geography.CorpusRetriever}.
 */
public final class GeographyCorpus implements Corpus {

    private static final List<String> SENTENCES = List.of(
        "Paris is the capital of France.",
        "France is a country in Europe.",
        "The Eiffel Tower is located in Paris.",
        "Tokyo is the capital of Japan.",
        "Japan is an island nation in East Asia.",
        "Mount Fuji is the highest mountain in Japan.",
        "Berlin is the capital of Germany.",
        "Germany is located in Central Europe.",
        "London is the capital of the United Kingdom.",
        "The United Kingdom is an island nation in Europe.",
        "Rome is the capital of Italy.",
        "Italy is famous for its art and history.",
        "Madrid is the capital of Spain.",
        "Spain is located on the Iberian Peninsula.",
        "Ottawa is the capital of Canada.",
        "Canada is the second largest country by area.",
        "Washington is the capital of the United States.",
        "The United States is located in North America.",
        "Brasilia is the capital of Brazil.",
        "Brazil is the largest country in South America.",
        "Canberra is the capital of Australia.",
        "Australia is both a country and a continent.",
        "New Delhi is the capital of India.",
        "India is located in South Asia.",
        "Beijing is the capital of China.",
        "China is the most populous country in the world.",
        "Moscow is the capital of Russia.",
        "Russia spans Europe and Asia.",
        "Cairo is the capital of Egypt.",
        "Egypt is located in North Africa.",
        "Nairobi is the capital of Kenya.",
        "Kenya is located in East Africa.",
        "The Nile is the longest river in Africa.",
        "The Amazon is the largest river by volume.",
        "The Amazon flows through Brazil and Peru.",
        "The Pacific Ocean is the largest ocean on Earth.",
        "The Atlantic Ocean separates Europe from the Americas.",
        "The Indian Ocean lies south of Asia.",
        "Mount Everest is the highest mountain in the world.",
        "Everest is located in the Himalayas.",
        "The Sahara is the largest hot desert in the world.",
        "The Arctic Ocean surrounds the North Pole.",
        "Antarctica is the coldest continent on Earth.",
        "The equator divides Earth into northern and southern hemispheres.",
        "The prime meridian passes through Greenwich in London.",
        "Asia is the largest continent by area and population.",
        "Africa is the second largest continent.",
        "Europe is known for its diverse cultures and history.",
        "South America is home to the Amazon rainforest.",
        "North America includes Canada the United States and Mexico.",
        "Oceania includes Australia New Zealand and Pacific islands.",
        "The Great Barrier Reef is located off the coast of Australia.",
        "The Grand Canyon is located in Arizona in the United States.",
        "The Rhine is a major river in Europe.",
        "The Danube flows through Central and Eastern Europe.",
        "Lake Baikal is the deepest lake in the world.",
        "Lake Victoria is the largest lake in Africa.",
        "The Andes are the longest mountain range in the world.",
        "The Rocky Mountains extend through North America.",
        "The Alps are a mountain range in Europe.",
        "Iceland is known for volcanoes and geothermal activity.",
        "Norway is famous for its fjords.",
        "Switzerland is a landlocked country in the Alps.",
        "Thailand is located in Southeast Asia.",
        "Vietnam shares a border with China.",
        "Indonesia is the largest archipelago in the world.",
        "Mexico City is the capital of Mexico.",
        "Argentina is located at the southern tip of South America.",
        "Chile has a long narrow shape along the Pacific coast.",
        "Colombia is located in northwestern South America.",
        "Peru is home to Machu Picchu.",
        "South Africa has three capital cities.",
        "Nigeria is the most populous country in Africa.",
        "Morocco is located in northwestern Africa.",
        "Saudi Arabia is located on the Arabian Peninsula.",
        "Iran is located in Western Asia.",
        "Turkey connects Europe and Asia.",
        "Greece is known as the cradle of Western civilization.",
        "Portugal is located on the western edge of Europe.",
        "Poland is located in Central Europe.",
        "Ukraine is the largest country entirely in Europe.",
        "Sweden is located in Northern Europe.",
        "Finland is known as the land of a thousand lakes.",
        "New Zealand is located southeast of Australia.",
        "Hawaii is a state in the Pacific Ocean.",
        "Greenland is the largest island in the world.",
        "Madagascar is the fourth largest island in the world.",
        "The Panama Canal connects the Atlantic and Pacific oceans.",
        "The Suez Canal connects the Mediterranean and Red seas.",
        "The English Channel separates England from France.",
        "The Mediterranean Sea borders Europe Africa and Asia.",
        "The Caribbean Sea is located near Central America.",
        "The Ganges is a sacred river in India.",
        "The Yangtze is the longest river in Asia.",
        "The Mississippi flows through the United States.",
        "The Volga is the longest river in Europe.",
        "The Congo River flows through Central Africa.",
        "The Mekong flows through Southeast Asia.",
        "The Dead Sea is the lowest point on land.",
        "The Mariana Trench is the deepest part of the ocean.",
        "The Ring of Fire is a zone of volcanic activity in the Pacific.",
        "The tropic of cancer lies north of the equator.",
        "The tropic of capricorn lies south of the equator."
    );

    private static final List<String> QUESTIONS = List.of(
        "What is the capital of France?",
        "What is the capital of Japan?",
        "Which continent is Brazil in?",
        "What is the longest river in Africa?",
        "What is the highest mountain in the world?",
        "Which ocean is the largest?",
        "What is the capital of Australia?",
        "Where is the Eiffel Tower located?",
        "What is the capital of India?",
        "Which country is the most populous?"
    );

    @Override
    public String name() {
        return "World Geography";
    }

    @Override
    public String description() {
        return "Factual sentences about countries, capitals, rivers, mountains, and oceans.";
    }

    @Override
    public List<String> sentences() {
        return SENTENCES;
    }

    @Override
    public List<String> sampleQuestions() {
        return QUESTIONS;
    }
}
