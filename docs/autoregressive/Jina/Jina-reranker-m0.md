---
sidebar_position: 1
---

# Jina Reranker m0

## 1. Model Introduction

[Jina Reranker m0](https://huggingface.co/jinaai/jina-reranker-m0) is Jina AI's multilingual reranking model designed to improve search relevance and document ranking. Built as a **Cross-Encoder architecture**, it processes query-document pairs together to produce direct relevance scores, making it ideal for reordering search results based on semantic relevance.

**Key Features:**

- **Cross-Encoder Architecture**: Processes query and document together for accurate relevance scoring
- **Multilingual Support**: Optimized for 89 languages with strong performance across diverse linguistic contexts
- **Extended Context**: 8192 token context window for processing long documents
- **Efficient Deployment**: Compact model size enables deployment on single AMD GPU
- **High Accuracy**: State-of-the-art reranking performance on standard benchmarks
- **Fast Inference**: Optimized for low-latency reranking in production environments

**Use Cases:**
- Search result reranking and relevance optimization
- Document retrieval and ranking in RAG systems
- Multi-stage ranking pipelines
- Query-document relevance scoring

**License:**
Jina Reranker m0 is licensed under the Apache 2.0 License. See [LICENSE](https://huggingface.co/jinaai/jina-reranker-m0) for details.

For more details, please refer to the [official Jina AI model page](https://huggingface.co/jinaai/jina-reranker-m0).

## 2. SGLang Installation

Please refer to the [official SGLang installation guide](https://docs.sglang.ai/get_started/install.html) for installation instructions.

## 3. Model Deployment

This section provides deployment configurations optimized for AMD GPUs (MI300X, MI325X, MI355X).

### 3.1 Interactive Configuration

**Interactive Command Generator**: Use the configuration selector below to automatically generate the appropriate deployment command for your AMD GPU setup.

import JinaRerankerConfigGenerator from '@site/src/components/autoregressive/JinaRerankerConfigGenerator';

<JinaRerankerConfigGenerator />

### 3.2 Configuration Tips

**AMD GPU Deployment:**
- All AMD GPUs (MI300X, MI325X, MI355X) support TP=1 deployment
- **Embedding Mode**: Uses `--is-embedding` flag for reranker inference in SGLang
- **Attention Backend**: Configured with `--attention-backend triton` for optimal performance
- **Cache Configuration**: Uses `--disable-radix-cache` as radix cache is not applicable for reranking tasks
- **Trust Remote Code**: Required flag `--trust-remote-code` for model initialization

**Performance Optimization:**
- The `--skip-server-warmup` flag reduces startup time
- Single GPU (TP=1) provides sufficient throughput for most reranking workloads
- For higher throughput requirements, consider deploying multiple instances with load balancing

## 4. Model Invocation

### 4.1 Basic Reranking Usage

Jina Reranker m0 is a Cross-Encoder that takes query-document pairs and outputs relevance scores directly. Use the `/v1/rerank` endpoint:

**Python Example:**

```python
import requests

# Define query and documents to rerank
query = "What is the capital of France?"
documents = [
    "Paris is the capital and most populous city of France.",
    "Berlin is the capital of Germany.",
    "London is the capital of the United Kingdom.",
    "Madrid is the capital of Spain.",
    "Rome is the capital of Italy."
]

# Call the rerank API
response = requests.post(
    "http://localhost:30000/v1/rerank",
    json={
        "model": "jinaai/jina-reranker-m0",
        "query": query,
        "documents": documents,
        "top_n": 5,
        "return_documents": True
    }
)

results = response.json()

print("Ranked Documents:")
for result in results["results"]:
    idx = result["index"]
    score = result["relevance_score"]
    doc = result["document"]["text"]
    print(f"{idx + 1}. [Score: {score:.4f}] {doc}")
```

**Expected Output:**
```
Ranked Documents:
1. [Score: 0.9876] Paris is the capital and most populous city of France.
2. [Score: 0.2341] Berlin is the capital of Germany.
3. [Score: 0.2198] Madrid is the capital of Spain.
4. [Score: 0.2156] London is the capital of the United Kingdom.
5. [Score: 0.2089] Rome is the capital of Italy.
```

### 4.2 Advanced Usage

#### 4.2.1 Batch Reranking

For processing multiple queries efficiently:

```python
import requests

def rerank_documents(query, documents, top_n=5):
    """Rerank documents based on query relevance using Cross-Encoder."""
    response = requests.post(
        "http://localhost:30000/v1/rerank",
        json={
            "model": "jinaai/jina-reranker-m0",
            "query": query,
            "documents": documents,
            "top_n": top_n,
            "return_documents": True
        }
    )

    if response.status_code != 200:
        raise Exception(f"Rerank API error: {response.text}")

    results = response.json()

    # Extract ranked documents and scores
    ranked_docs = []
    scores = []
    for result in results["results"]:
        ranked_docs.append(result["document"]["text"])
        scores.append(result["relevance_score"])

    return ranked_docs, scores

# Example usage
query = "How to deploy machine learning models?"
documents = [
    "Deploy ML models using Docker containers for isolation.",
    "Python is a popular programming language.",
    "Use Kubernetes for orchestrating model deployments.",
    "Machine learning requires large datasets for training.",
    "SGLang provides efficient LLM inference capabilities."
]

ranked_docs, scores = rerank_documents(query, documents, top_n=3)

print(f"Query: {query}\n")
print("Top Reranked Results:")
for i, (doc, score) in enumerate(zip(ranked_docs, scores), 1):
    print(f"{i}. [Score: {score:.4f}] {doc}")
```

#### 4.2.2 Integration with RAG Systems

Use Jina Reranker to improve retrieval quality in RAG pipelines:

```python
import requests
import openai

# Initialize LLM client
llm_client = openai.OpenAI(
    base_url="http://localhost:30001/v1",  # LLM endpoint
    api_key="EMPTY"
)

def rag_with_reranking(query, candidate_documents, top_k=3):
    """RAG pipeline with reranking step."""

    # Step 1: Rerank candidate documents using Cross-Encoder
    rerank_response = requests.post(
        "http://localhost:30000/v1/rerank",
        json={
            "model": "jinaai/jina-reranker-m0",
            "query": query,
            "documents": candidate_documents,
            "top_n": top_k,
            "return_documents": True
        }
    )

    if rerank_response.status_code != 200:
        raise Exception(f"Rerank API error: {rerank_response.text}")

    rerank_results = rerank_response.json()

    # Step 2: Extract top-k documents
    top_docs = [result["document"]["text"] for result in rerank_results["results"]]

    # Step 3: Generate response with LLM
    context = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(top_docs)])
    prompt = f"""Based on the following context, answer the question.

Context:
{context}

Question: {query}

Answer:"""

    completion = llm_client.chat.completions.create(
        model="meta-llama/Llama-3.3-70B-Instruct",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=500
    )

    return completion.choices[0].message.content, top_docs

# Example usage
query = "What are the benefits of using Docker?"
documents = [
    "Docker provides containerization for applications, ensuring consistency across environments.",
    "Kubernetes is an orchestration platform for containers.",
    "Docker enables microservices architecture by isolating application components.",
    "Python is a versatile programming language used in data science.",
    "Container images are lightweight and portable, making deployment easier.",
    "Docker Hub is a registry for sharing container images."
]

answer, relevant_docs = rag_with_reranking(query, documents, top_k=3)

print(f"Query: {query}\n")
print(f"Answer: {answer}\n")
print("Top Relevant Documents:")
for i, doc in enumerate(relevant_docs, 1):
    print(f"{i}. {doc}")
```

## 5. Benchmarking

Use the SGLang benchmarking suite to test reranking performance:

### 5.1 Basic Benchmark Command

```bash
python -m sglang.bench_serving \
  --backend sglang \
  --host localhost \
  --port 30000 \
  --dataset-name random \
  --num-prompts 500 \
  --random-input 512 \
  --random-output 1 \
  --max-concurrency 32
```

### 5.2 Adjusting Benchmark Parameters

**Input Length**: Adjust `--random-input` to test different document sizes:
- Short documents: `--random-input 256`
- Medium documents: `--random-input 512`
- Long documents: `--random-input 1024`

**Concurrency Levels**: Adjust `--max-concurrency` to test different load scenarios:
- Low concurrency (latency-focused): `--max-concurrency 4 --num-prompts 100`
- Medium concurrency (balanced): `--max-concurrency 32 --num-prompts 500`
- High concurrency (throughput-focused): `--max-concurrency 128 --num-prompts 1000`

**Note**:
- Always specify `--host localhost` and `--port 30000` to connect to the correct endpoint
- For reranking models, `--random-output 1` is recommended since reranking produces scores rather than generated text
- The benchmark measures throughput for query-document pair processing

---

## 📚 Additional Resources

- [Jina Reranker Model Card](https://huggingface.co/jinaai/jina-reranker-m0)
- [Jina AI Documentation](https://jina.ai/reranker/)
- [SGLang Documentation](https://docs.sglang.ai/)
- [AMD ROCm Documentation](https://rocm.docs.amd.com/)
