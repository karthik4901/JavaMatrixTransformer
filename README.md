# JavaMatrixTransformer

A from-scratch transformer implementation in Java 21, with a Spring Boot API for geography Q&A.

Most transformer tutorials live in Python. This project rebuilds the core pieces — matrix math, tokenization, embeddings, attention, and a transformer block — in plain Java so the mechanics are easy to read and step through. On top of that, it exposes a small geography question-answering service over HTTP and the terminal.

This is a learning and engineering project, not a production LLM. Model weights are randomly initialized. Geography answers today come from corpus retrieval over a fixed fact set, not from trained next-token generation.

**Author:** Karthik Goud · [GitHub](https://github.com/karthik4901)

---

## Why this exists

Three problems I wanted to solve for myself:

1. **Understand transformers without a framework.** Reading PyTorch code hides the math. Writing multiply, softmax, and attention by hand makes the paper concrete.
2. **Keep the JVM in the loop.** Java engineers often treat ML as a Python-only skill. This repo shows the same architecture can live on the JVM.
3. **Ship something usable early.** A demo that only prints matrices is hard to share. A Spring Boot endpoint and a terminal chat make the project testable and reviewable.

Geography was chosen as the first domain because facts are short, checkable, and easy to expand later.

---

## Features

- Pure Java matrix engine (multiply, transpose, softmax, layer norm, cosine similarity)
- Word-level tokenizer with special tokens (`PAD`, `UNK`, `BOS`, `EOS`)
- Embedding layer and sine/cosine positional encoding
- Scaled dot-product attention and multi-head attention
- Transformer block with residual connections and feed-forward network
- Composable `TransformerModel` pipeline
- Pluggable `Corpus` / `Tokenizer` interfaces
- Geography Q&A via keyword retrieval over 100+ facts
- Spring Boot REST API (`/health`, `/api/ask`)
- Interactive terminal chat
- Internals walkthrough demo
- Controller tests with MockMvc

---

## Architecture

```
Text
  → Tokenizer          (words → token IDs)
  → EmbeddingLayer     (IDs → dense vectors)
  → PositionalEncoding (add position signal)
  → TransformerBlock×N (attention + FFN + residuals)
  → LM head            (context → next-token logits)
```

Geography Q&A path (current):

```
HTTP / chat
  → GeographyAssistant
  → CorpusRetriever     (keyword score against GeographyCorpus)
  → best matching fact
```

The transformer stack is built and held by `GeographyAssistant` so the pipeline is real and inspectable. Answers are retrieved from the corpus until training is implemented. That split is intentional: architecture first, training later.

### Package layout

```
com.karthik.transformer
├── JavaMatrixTransformerApplication   Spring Boot entry point
├── spring/                            Bean wiring
├── web/                               REST controller + DTOs
├── config/                            ModelConfig
├── core/                              Matrix
├── tokenizer/                         Tokenizer, SimpleTokenizer
├── embedding/                         EmbeddingLayer
├── encoding/                          PositionalEncoding
├── attention/                         Attention + TransformerBlock
├── model/                             TransformerModel
├── data/                              Corpus, GeographyCorpus
├── geography/                         Assistant, retriever, chat CLI
└── demo/                              Internals and batch demos
```

### Core formula

```
Attention(Q, K, V) = softmax(Q × Kᵀ / √dₖ) × V
```

Implemented in `ScaledDotProductAttention`.

---

## Requirements

- Java 21+
- Gradle 8+ (wrapper included)

---

## Setup

```bash
git clone https://github.com/karthik4901/JavaMatrixTransformer.git
cd JavaMatrixTransformer
./gradlew build
```

---

## Run

### Spring Boot API

```bash
./gradlew bootRun
```

| Item | Value |
|------|-------|
| Main class | `com.karthik.transformer.JavaMatrixTransformerApplication` |
| Port | `8080` (`src/main/resources/application.properties`) |

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

Example response:

```json
{
  "question": "What is the capital of France?",
  "answer": "Paris is the capital of France."
}
```

### Terminal chat

```bash
./gradlew chat
```

```
Question: What is the longest river in Africa?
Answer:  The Nile is the longest river in Africa.
```

### Internals walkthrough

Prints each building block with geography examples:

```bash
./gradlew demo
```

### Batch Q&A

Runs the sample question list from `GeographyCorpus`:

```bash
./gradlew geography
```

### Command summary

| Command | What it does |
|---------|--------------|
| `./gradlew build` | Compile and run tests |
| `./gradlew bootRun` | Start Spring Boot API on :8080 |
| `./gradlew demo` | Transformer internals demo |
| `./gradlew geography` | Batch geography Q&A |
| `./gradlew chat` | Interactive terminal chat |
| `./gradlew clean build` | Clean rebuild |

---

## Testing

```bash
./gradlew test
```

Current coverage focuses on the HTTP layer:

- `GeographyControllerTest` — health, GET ask, POST ask (MockMvc + mocked assistant)

### TODO — tests still needed

- [ ] Unit tests for `Matrix` (shape checks, softmax row sums, layer norm)
- [ ] Unit tests for `SimpleTokenizer` (encode/decode round-trip, unknown tokens)
- [ ] Unit tests for `CorpusRetriever` (exact match, weak match, empty query)
- [ ] Unit tests for `TransformerModel.forward` (shape invariants)
- [ ] Integration test for `bootRun` health endpoint without mocking

---

## Engineering notes

**Why pure Java matrices?**  
No ND4J or PyTorch JNI. Every operation is readable. That makes the project useful for interviews and teaching, at the cost of speed.

**Why word-level tokenization?**  
BPE is the right production choice. Word-level keeps the first version inspectable. The `Tokenizer` interface is there so BPE can be added without rewriting the model.

**Why retrieval for Q&A?**  
Untrained random weights produce garbage text. Returning a scored corpus fact is honest and useful. The transformer path stays in the codebase for the next step: training.

**Why Spring Boot for a teaching model?**  
A REST surface makes the project reviewable by other engineers. It also forces clean boundaries: controller → assistant → retriever/model.

**Config as a record.**  
`ModelConfig` validates `dModel % numHeads == 0` at construction. Invalid shapes fail fast instead of failing deep inside attention.

**Performance awareness.**  
Attention is O(n²) in sequence length. The demo config uses `dModel=16`, one layer, and short sequences on purpose. Scaling needs batching, better matrix kernels, and eventually native acceleration — none of that is claimed here.

---

## Future improvements

1. **Training loop** — cross-entropy loss, backprop through the transformer, Adam optimizer, checkpoint save/load
2. **BPE tokenizer** — replace word-level tokenization for better coverage
3. **Larger corpus** — more geography facts, then other domains via `Corpus`
4. **Neural answers** — once trained, generate answers from the LM head instead of retrieval
5. **KV cache** — speed up autoregressive decoding
6. **Broader tests** — matrix, tokenizer, retriever, and model shape tests listed above
7. **Write-up** — short technical note or blog post on implementing attention in Java

---

## Honest limits

| Claim | Reality |
|-------|---------|
| “Transformer from scratch” | Yes — core math and layers are hand-written |
| “Trained language model” | No — weights are random |
| “Geography Q&A” | Yes — via corpus retrieval |
| “Production LLM” | No |
| “Research paper ready” | Not yet — training and evaluation still needed |

---

## License

MIT
