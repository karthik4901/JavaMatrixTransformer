# JavaMatrixTransformer

**Transformer internals from scratch in Java 21 — plus a Spring Boot geography Q&A API.**

> Learn transformers the same way you learned Java: Hello World → variables → methods → your first real project.

This is **not** a billion-parameter LLM. It is a small teaching repo that implements the transformer architecture behind GPT and BERT — matrix by matrix — with a working **World Geography** Q&A app on top.

## Prerequisites

```bash
java --version   # 21+
gradle --version # 8+
```

## Quick Start

```bash
git clone https://github.com/karthikgoud/JavaMatrixTransformer.git
cd JavaMatrixTransformer
gradle build
gradle bootRun          # Spring Boot REST API → http://localhost:8080
```

## Run & Test

### Build and test

```bash
gradle build
```

Runs unit tests (including Spring MVC controller tests). Expect `BUILD SUCCESSFUL`.

### Spring Boot REST API (main entry point)

```bash
gradle bootRun
```

| Setting | Value |
|---------|-------|
| Main class | `com.karthik.transformer.JavaMatrixTransformerApplication` |
| Port | `8080` (`src/main/resources/application.properties`) |
| Framework | Spring Boot 3 |

**Endpoints:**

```bash
# Health
curl http://localhost:8080/health

# Ask (GET)
curl "http://localhost:8080/api/ask?q=What+is+the+capital+of+France"

# Ask (POST)
curl -X POST http://localhost:8080/api/ask \
  -H "Content-Type: application/json" \
  -d '{"question":"What is the capital of Japan?"}'
```

**Example response:**

```json
{"question":"What is the capital of France?","answer":"Paris is the capital of France."}
```

### Transformer internals demo

Walks through all 9 building blocks (matrix → tokenization → attention → transformer):

```bash
gradle demo
```

### Batch geography Q&A

Runs 10 sample questions automatically:

```bash
gradle geography
```

### Interactive terminal chat

```bash
gradle chat
```

```
────────────────────────────────────────
Question: What is the capital of France?
Answer:  Paris is the capital of France.

────────────────────────────────────────
Question: exit
Goodbye!
```

## Command Reference

| Command | Description |
|---------|-------------|
| `gradle build` | Compile + test |
| `gradle bootRun` | **Spring Boot API** (port 8080) |
| `gradle demo` | Internals walkthrough |
| `gradle geography` | Batch Q&A (10 questions) |
| `gradle chat` | Interactive terminal chat |
| `gradle clean build` | Clean rebuild |

## Project Structure

```
com.karthik.transformer
├── JavaMatrixTransformerApplication.java   ← Spring Boot main
├── spring/          GeographyBeanConfig
├── web/             GeographyController + DTOs
├── config/          ModelConfig
├── core/            Matrix
├── tokenizer/       Tokenizer → SimpleTokenizer
├── embedding/       EmbeddingLayer
├── encoding/        PositionalEncoding
├── attention/       ScaledDotProductAttention → TransformerBlock
├── model/           TransformerModel
├── data/            Corpus → GeographyCorpus
├── geography/       GeographyAssistant, CorpusRetriever, GeographyChat
└── demo/            Demo, GeographyQADemo, Main (non-Spring CLI)
```

## What's Implemented

| Layer | Class | Purpose |
|-------|-------|---------|
| Matrix engine | `Matrix` | Multiply, transpose, softmax, layer norm |
| Tokenizer | `Tokenizer` / `SimpleTokenizer` | Text ↔ token IDs |
| Embeddings | `EmbeddingLayer` | Token ID → dense vectors |
| Position | `PositionalEncoding` | Sine/cosine position injection |
| Attention | `ScaledDotProductAttention` | Core attention mechanism |
| Multi-head | `MultiHeadAttention` | Parallel attention heads |
| Transformer | `TransformerBlock` | Full encoder layer |
| Model | `TransformerModel` | Composable transformer pipeline |
| Geography | `GeographyAssistant` | Q&A via corpus retrieval |
| API | `GeographyController` | Spring Boot REST endpoints |

## How Geography Q&A Works

1. **Transformer demo** — `gradle demo` shows how the neural architecture works (untrained weights).
2. **Geography answers** — chat and API use **corpus retrieval**: keyword matching against 100+ geography facts.
3. **Future** — backprop training would let the model generate answers directly.

## Key Formula

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √dₖ) × V
```

## Why Not Call It an "LLM"?

A real LLM has billions of trained parameters. This repo teaches **how the transformer machinery works** at small scale. That is honest and useful.

## Author

**Karthik Goud** — [GitHub](https://github.com/karthikgoud)

## License

MIT — use freely, attribution appreciated.
