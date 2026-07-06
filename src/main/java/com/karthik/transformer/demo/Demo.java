package com.karthik.transformer.demo;

import com.karthik.transformer.attention.MultiHeadAttention;
import com.karthik.transformer.attention.ScaledDotProductAttention;
import com.karthik.transformer.attention.TransformerBlock;
import com.karthik.transformer.config.ModelConfig;
import com.karthik.transformer.core.Matrix;
import com.karthik.transformer.data.GeographyCorpus;
import com.karthik.transformer.embedding.EmbeddingLayer;
import com.karthik.transformer.encoding.PositionalEncoding;
import com.karthik.transformer.tokenizer.SimpleTokenizer;

import java.util.List;

/**
 * End-to-end walkthrough of every LLM building block — using world geography examples.
 *
 * Run: {@code gradle demo}
 */
public final class Demo {

    private static final ModelConfig CONFIG = ModelConfig.DEMO;
    private static final GeographyCorpus CORPUS = new GeographyCorpus();

    public static void main(String[] args) {
        printBanner();
        demo1_Tokenization();
        demo2_MatrixOps();
        demo3_Embeddings();
        demo4_PositionalEncoding();
        demo5_ScaledDotProductAttention();
        demo6_MultiHeadAttention();
        demo7_TransformerBlock();
        demo8_FullPipeline();
        demo9_ModelScale();
        printFooter();
    }

    static void demo1_Tokenization() {
        section("1. TOKENIZATION — Text → Token IDs");

        SimpleTokenizer tokenizer = new SimpleTokenizer();
        tokenizer.buildVocabulary(CORPUS.sentences());
        System.out.println(tokenizer);

        String sentence = "paris is the capital of france";
        int[] encoded = tokenizer.encode(sentence);
        String decoded = tokenizer.decode(encoded);

        System.out.println("\nInput:   \"" + sentence + "\"");
        System.out.println("Encoded: " + formatIds(encoded));
        System.out.println("Decoded: \"" + decoded + "\"");

        System.out.println("\nVocabulary sample:");
        for (String word : List.of("paris", "france", "tokyo", "japan", "capital", "europe")) {
            if (tokenizer.hasToken(word)) {
                System.out.printf("  %-12s → ID %d%n", word, tokenizer.getId(word));
            }
        }
    }

    static void demo2_MatrixOps() {
        section("2. MATRIX OPERATIONS — The Math Behind Everything");

        double[] paris  = {0.9, 0.8, 0.3}; // culture, history, population
        double[] tokyo  = {0.85, 0.75, 0.95};
        double[] berlin = {0.7, 0.85, 0.5};

        System.out.println("City vectors (culture, history, population):");
        System.out.printf("  Paris:  %s%n", vecStr(paris));
        System.out.printf("  Tokyo:  %s%n", vecStr(tokyo));
        System.out.printf("  Berlin: %s%n", vecStr(berlin));

        System.out.println("\nDot products (raw similarity):");
        System.out.printf("  Paris · Tokyo  = %.4f%n", Matrix.dotProduct(paris, tokyo));
        System.out.printf("  Paris · Berlin = %.4f%n", Matrix.dotProduct(paris, berlin));
        System.out.printf("  Tokyo · Berlin = %.4f%n", Matrix.dotProduct(tokyo, berlin));

        System.out.println("\nCosine similarity (normalized, -1 to +1):");
        System.out.printf("  Paris ↔ Tokyo  = %.4f%n", Matrix.cosineSimilarity(paris, tokyo));
        System.out.printf("  Paris ↔ Berlin = %.4f%n", Matrix.cosineSimilarity(paris, berlin));

        System.out.println("\nMatrix multiplication: (2×3) × (3×2) = (2×2)");
        Matrix a = new Matrix(new double[][]{{1, 2, 3}, {4, 5, 6}});
        Matrix b = new Matrix(new double[][]{{7, 8}, {9, 10}, {11, 12}});
        System.out.print("A × B = " + a.multiply(b));

        System.out.println("Softmax (raw scores → probabilities):");
        Matrix scores = new Matrix(new double[][]{{2.0, 1.0, 0.1}});
        Matrix probs = scores.softmax();
        System.out.print("  Scores: " + scores);
        System.out.print("  Probs:  " + probs);
        double rowSum = probs.get(0, 0) + probs.get(0, 1) + probs.get(0, 2);
        System.out.printf("  Sum of probabilities: %.6f (always = 1.0)%n", rowSum);
    }

    static void demo3_Embeddings() {
        section("3. EMBEDDING LAYER — Token IDs → Dense Vectors");

        SimpleTokenizer tokenizer = new SimpleTokenizer();
        tokenizer.buildVocabulary(CORPUS.sentences().subList(0, 10));

        EmbeddingLayer embedding = new EmbeddingLayer(tokenizer.vocabularySize(), CONFIG.dModel());
        System.out.println(embedding);

        int parisId = tokenizer.getId("paris");
        int tokyoId = tokenizer.getId("tokyo");

        System.out.printf("%nEmbedding for 'paris' (ID=%d): first 6 dims: %s%n",
            parisId, vecStr6(embedding.lookup(parisId)));
        System.out.printf("Embedding for 'tokyo' (ID=%d): first 6 dims: %s%n",
            tokyoId, vecStr6(embedding.lookup(tokyoId)));

        System.out.println("\nNote: These are random — in a trained model,");
        System.out.println("related words (e.g. paris & france) would have similar vectors.");

        int[] ids = tokenizer.encode("paris is the capital of france");
        Matrix embedded = embedding.embed(ids);
        System.out.printf("%nEmbedded sequence shape: %d tokens × %d dims%n",
            embedded.rows, embedded.cols);
    }

    static void demo4_PositionalEncoding() {
        section("4. POSITIONAL ENCODING — Where is Each Token?");

        PositionalEncoding pe = new PositionalEncoding(CONFIG.maxSeqLength(), CONFIG.dModel());
        System.out.println(pe);

        System.out.println("\nPositional encodings (first 4 dims) for positions 0–4:");
        System.out.printf("  %-6s  %-8s  %-8s  %-8s  %-8s%n", "Pos", "Dim0", "Dim1", "Dim2", "Dim3");
        for (int pos = 0; pos <= 4; pos++) {
            double[] enc = pe.getEncoding(pos);
            System.out.printf("  %-6d  %-8.4f  %-8.4f  %-8.4f  %-8.4f%n",
                pos, enc[0], enc[1], enc[2], enc[3]);
        }

        System.out.println("\nPosition similarity (nearby = more similar):");
        System.out.printf("  pos0 ↔ pos1: %.4f%n", pe.positionSimilarity(0, 1));
        System.out.printf("  pos0 ↔ pos5: %.4f%n", pe.positionSimilarity(0, 5));
        System.out.printf("  pos0 ↔ pos20: %.4f%n", pe.positionSimilarity(0, 20));
    }

    static void demo5_ScaledDotProductAttention() {
        section("5. SCALED DOT-PRODUCT ATTENTION — The Core Mechanism");

        System.out.println("Formula: Attention(Q, K, V) = softmax(Q × K^T / sqrt(dK)) × V%n");

        int seqLen = 4;
        String[] tokens = {"paris", "is", "the", "capital"};

        ScaledDotProductAttention attention = new ScaledDotProductAttention(
            CONFIG.dModel(), CONFIG.headDim(), CONFIG.headDim());

        Matrix input = Matrix.random(seqLen, CONFIG.dModel(), 0.1);
        Matrix output = attention.forward(input);

        System.out.printf("Input shape:  %d × %d%n", input.rows, input.cols);
        System.out.printf("Output shape: %d × %d%n", output.rows, output.cols);

        Matrix weights = attention.getLastAttentionWeights();
        System.out.println("\nAttention weights (row i attends to column j):");
        printWeightMatrix(tokens, weights, seqLen);
        System.out.println("(Each row sums to 1.0 — probability distributions)");

        System.out.println("\nMasked attention (autoregressive — cannot see the future):");
        attention.maskedForward(input);
        printWeightMatrix(tokens, attention.getLastAttentionWeights(), seqLen);
        System.out.println("(Upper triangle = 0 — future tokens are masked)");
    }

    static void demo6_MultiHeadAttention() {
        section("6. MULTI-HEAD ATTENTION — Parallel Attention Patterns");

        System.out.printf("Running %d attention heads in parallel, each dim=%d%n%n",
            CONFIG.numHeads(), CONFIG.headDim());

        MultiHeadAttention mha = new MultiHeadAttention(CONFIG.dModel(), CONFIG.numHeads());
        System.out.println(mha);

        Matrix input = Matrix.random(4, CONFIG.dModel(), 0.1);
        Matrix output = mha.forward(input);
        System.out.printf("%nInput:  %d × %d%n", input.rows, input.cols);
        System.out.printf("Output: %d × %d%n", output.rows, output.cols);

        System.out.println("\nEach head can learn different patterns:");
        System.out.println("  Head 1 → grammatical roles (subject–verb)");
        System.out.println("  Head 2 → semantic similarity (paris ↔ france)");
        System.out.println("  Head 3 → coreference (it → the country)");
        System.out.println("  Head 4 → positional proximity");
    }

    static void demo7_TransformerBlock() {
        section("7. TRANSFORMER BLOCK — One Complete Encoder Layer");

        System.out.println("Architecture: Attention → Add&Norm → FeedForward → Add&Norm%n");

        TransformerBlock block = new TransformerBlock(
            CONFIG.dModel(), CONFIG.numHeads(), CONFIG.dFF());
        System.out.println(block);

        Matrix input = Matrix.random(5, CONFIG.dModel(), 0.1);
        Matrix output = block.forward(input);
        System.out.printf("%nInput:  %d × %d%n", input.rows, input.cols);
        System.out.printf("Output: %d × %d%n", output.rows, output.cols);
        System.out.println("Stack 12 blocks → BERT-base. Stack 96 → GPT-3.");
    }

    static void demo8_FullPipeline() {
        section("8. FULL PIPELINE — Text to Contextual Representations");

        String sentence = "paris is the capital of france";
        System.out.println("Input sentence: \"" + sentence + "\"\n");

        SimpleTokenizer tokenizer = new SimpleTokenizer();
        tokenizer.buildVocabulary(CORPUS.sentences());

        int[] tokenIds = tokenizer.encode(sentence);
        System.out.printf("Step 1 — Tokenize: %d tokens%n", tokenIds.length);

        EmbeddingLayer embedding = new EmbeddingLayer(tokenizer.vocabularySize(), CONFIG.dModel());
        Matrix embedded = embedding.embed(tokenIds);
        System.out.printf("Step 2 — Embed: %d × %d matrix%n", embedded.rows, embedded.cols);

        PositionalEncoding pe = new PositionalEncoding(CONFIG.maxSeqLength(), CONFIG.dModel());
        Matrix withPos = pe.addPositionalEncoding(embedded);
        System.out.printf("Step 3 — Add position: %d × %d%n", withPos.rows, withPos.cols);

        TransformerBlock block = new TransformerBlock(
            CONFIG.dModel(), CONFIG.numHeads(), CONFIG.dFF());
        Matrix output = block.forward(withPos);
        System.out.printf("Step 4 — Transformer: %d × %d contextual output%n",
            output.rows, output.cols);

        System.out.println("\nEach row is now a context-aware representation of that token.");
    }

    static void demo9_ModelScale() {
        section("9. MODEL SCALE — Parameters at Different Sizes");

        System.out.println("Our demo model vs production LLMs:\n");
        System.out.printf("  %-20s  %-8s  %-6s  %-5s  %-12s%n",
            "Model", "dModel", "Heads", "Layers", "Parameters");
        System.out.println("  " + "─".repeat(60));

        printModelStats("This demo", CONFIG.dModel(), CONFIG.numHeads(), 1, CONFIG.dFF());
        printModelStats("GPT-1", 768, 12, 12, 3072);
        printModelStats("BERT-base", 768, 12, 12, 3072);
        printModelStats("GPT-3", 12288, 96, 96, 49152);
    }

    static void printModelStats(String name, int dModel, int heads, int layers, int dFF) {
        long attnParams = (long) 4 * dModel * dModel;
        long ffnParams = (long) 2 * dModel * dFF;
        long total = (attnParams + ffnParams) * layers;
        String readable = total > 1_000_000_000L
            ? String.format("%.1fB", total / 1e9)
            : total > 1_000_000L
            ? String.format("%.1fM", total / 1e6)
            : String.format("%dK", total / 1000);
        System.out.printf("  %-20s  %-8d  %-6d  %-5d  %-12s%n",
            name, dModel, heads, layers, readable);
    }

    static void printWeightMatrix(String[] tokens, Matrix weights, int seqLen) {
        System.out.printf("  %-10s", "");
        for (int j = 0; j < seqLen; j++) System.out.printf("%-10s", tokens[j]);
        System.out.println();
        for (int i = 0; i < seqLen; i++) {
            System.out.printf("  %-10s", tokens[i]);
            for (int j = 0; j < seqLen; j++) {
                System.out.printf("%-10.4f", weights.get(i, j));
            }
            System.out.println();
        }
    }

    static void section(String title) {
        System.out.println("\n" + "═".repeat(60));
        System.out.println("  " + title);
        System.out.println("═".repeat(60) + "\n");
    }

    static void printBanner() {
        System.out.println("╔══════════════════════════════════════════════════════════╗");
        System.out.println("║     JavaMatrixTransformer — Built From Scratch in Java   ║");
        System.out.println("║   Tokenization → Embedding → Attention → Transformer     ║");
        System.out.println("║   Domain: World Geography · Author: Karthik Goud         ║");
        System.out.println("╚══════════════════════════════════════════════════════════╝");
    }

    static void printFooter() {
        System.out.println("\n" + "═".repeat(60));
        System.out.println("  Pure Java 21 · Zero ML frameworks · Open source (MIT)");
        System.out.println("  Next: gradle geography  (batch Q&A)  |  gradle bootRun  (REST API)");
        System.out.println("═".repeat(60));
    }

    static String formatIds(int[] ids) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < ids.length; i++) {
            sb.append(ids[i]);
            if (i < ids.length - 1) sb.append(", ");
        }
        return sb.append("]").toString();
    }

    static String vecStr(double[] v) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < v.length; i++) {
            sb.append(String.format("%.2f", v[i]));
            if (i < v.length - 1) sb.append(", ");
        }
        return sb.append("]").toString();
    }

    static String vecStr6(double[] v) {
        StringBuilder sb = new StringBuilder("[");
        int lim = Math.min(6, v.length);
        for (int i = 0; i < lim; i++) {
            sb.append(String.format("%.4f", v[i]));
            if (i < lim - 1) sb.append(", ");
        }
        if (v.length > 6) sb.append(", ...");
        return sb.append("]").toString();
    }
}
