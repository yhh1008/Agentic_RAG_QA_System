# Agentic_RAG_QA_System

An agentic question-answering system built on a ReAct-like decision loop with `LangGraph + LangChain`, focused on Chinese school-policy QA.

## Features

- Agentic RAG workflow: rewrite/keyword -> retrieve -> rerank -> selective read -> answer with citations
- Vector retrieval with Chroma and metadata-complete chunks
- Keyword-aware lightweight reranking
- Structured tool I/O via Pydantic models
- Trace logging (`JSONL`) for SFT/RL trajectory generation
- FastAPI endpoint: `POST /ask`

## Project Structure

- `src/config/settings.py`: env and hyperparameters
- `src/llm/deepseek_client.py`: DeepSeek model client + structured invoke
- `src/retrieval/*`: embedding, vectorstore, reranking
- `src/agent/*`: prompts, tools, state, LangGraph workflow
- `src/api/app.py`: API server
- `src/eval/metrics.py`: baseline metrics
- `scripts/build_index.py`: data ingestion and indexing
- `scripts/run_local_chat.py`: local CLI chat
- `train/sft/*`, `train/rl/*`: training data + skeleton scripts

## Quickstart

1. Create env and install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

2. Configure environment

```bash
cp .env.example .env
# fill DEEPSEEK_API_KEY and optional overrides
```

3. Put knowledge files in `data/raw/` (`.pdf`, `.md`, `.txt`) and build index

```bash
python scripts/build_index.py
```

4. Run API

```bash
uvicorn src.api.app:app --reload --host 0.0.0.0 --port 8000
```

5. Test API

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query":"学生考试作弊怎么处理？"}'
```

## Training Pipeline (skeleton)

1. Generate trajectory-linked SFT data

```bash
python train/sft/build_sft_data.py
```

2. SFT entry

```bash
python train/sft/train_sft.py
```

3. RL/GRPO entry

```bash
python train/rl/train_grpo.py
```

## Notes

- Max retry attempts and rerank bonus are controlled via `.env`.
- Trace files are saved in `train/data/traces/*.jsonl`.
- Current SFT/RL scripts are intentionally minimal scaffolds for your preferred trainer stack (TRL/PEFT/VeRL).
