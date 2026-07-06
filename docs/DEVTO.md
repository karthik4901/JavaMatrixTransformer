# dev.to Article

**Title:** I Built a Transformer in Java — With a Spring Boot Geography API

**Tags:** `java`, `springboot`, `machinelearning`, `transformer`

**Repo:** https://github.com/karthikgoud/JavaMatrixTransformer

---

## Introduction

Learn transformers like you learned Java: Hello World → variables → methods → first project.

**JavaMatrixTransformer** implements attention and transformer blocks in pure Java (no PyTorch), then exposes a **Spring Boot REST API** for geography Q&A.

## Run It

```bash
gradle build
gradle bootRun
curl "http://localhost:8080/api/ask?q=What+is+the+capital+of+France"
```

## Architecture

- `Matrix`, `SimpleTokenizer`, `EmbeddingLayer`, `TransformerBlock` — from scratch
- `JavaMatrixTransformerApplication` — Spring Boot entry point
- `GeographyController` — `/health` and `/api/ask`
- `GeographyAssistant` — answers from 100+ geography facts

## Key Takeaway

> Teach the transformer honestly at small scale. Ship a real API on top.

---
