# 🧠 RAG Persona & Orchestration Pipeline

A Retrieval-Augmented Generation (RAG) system that routes queries between vector databases and fine-tuned models to maintain specific group chat personas and "vibe"

## 🛠 Project Architecture

The system employs a 3-tier routing logic to determine the most accurate data source and generation style for any given user query.

```mermaid
graph TD
    User([User Query]) --> Expand[Multi-Query Expansion]
    Expand --> Orchestrator{Orchestrator: <br/>Match Probability}
    
    Orchestrator -->|High DB Match| Route1[Route 1: MongoDB Vector RAG]
    Orchestrator -->|High Vibe Match| Route2[Route 2: Fine-Tuned Llama-3]
    Orchestrator -->|Complex/Hybrid| Route3[Route 3: Mixed Approach]

    subgraph R1 [Vectorized MongoDB]
    Route1 --> R1_Search[768D Vector Search]
    R1_Search --> R1_Filt[Base Threshold Filter]
    R1_Filt --> R1_Gen[Gemini 2.5 Flash-Lite Response]
    end

    subgraph R2 [Fine-Tuned Persona]
    Route2 --> R2_Search[Top-1 Context Retrieval]
    R2_Search --> R2_Gen[Llama-3 FT Generation]
    end

    subgraph R3 [Mixed Synthesis]
    Route3 --> R3_Initial[Gemini Initial Answer]
    R3_Initial --> R3_Sum[Context Summarization]
    R3_Sum --> R3_Refine[Llama-3 FT Persona Translation]
    end

## 🔄 Core Process
1. **Multi-Query Expansion:** Increases recall by generating semantic variations of the user's query.
2. **Probability-Based Routing:** Uses an LLM classifier to send the query to the best-fit data source.
3. **Hybrid Generation:**
    - **MongoDB RAG:** High-accuracy factual retrieval.
    - **FT-Llama:** Direct persona mimicry.
    - **Mixed:** Fact extraction via Gemini followed by Persona translation via Llama.

## ⚙️ Installation
```bash
pip install -r requirements.txt
