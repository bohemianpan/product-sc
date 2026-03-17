# MiroFish — Multi-Agent Simulation Prediction Engine

MiroFish takes seed materials (news, reports, policies) + a prediction question,
extracts a "world model", then runs multi-agent simulations to produce prediction reports.

## P0 Module — Document → World Model Pipeline

This module covers:
1. **Document Parser** — PDF/TXT/MD → plain text
2. **Chunker** — Split text into overlapping chunks with unique IDs
3. **Extractor** — LLM-powered extraction of actors, relationships, timeline, and variables

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
# Edit .env with your OpenAI API key

# Run the server
uvicorn app.main:app --reload --port 8000

# API docs
open http://localhost:8000/docs
```

## API Endpoints

- `POST /api/v1/documents/upload` — Upload a document (PDF/TXT/MD), returns parsed text + document_id
- `GET /api/v1/documents/{document_id}/chunks` — Get chunks for a document
- `POST /api/v1/documents/{document_id}/extract` — Extract world model from document given a prediction question

## Testing

```bash
# Run the Nongfu Spring integration test (uses mock LLM by default)
python -m tests.test_nongfu

# Or with real GPT-4o (requires OPENAI_API_KEY in .env)
USE_MOCK_LLM=false python -m tests.test_nongfu
```

## Project Structure

```
mirofish/
├── README.md
├── requirements.txt
├── .env.example
├── app/
│   ├── main.py          # FastAPI application
│   ├── config.py         # Settings (env-based)
│   ├── models/
│   │   ├── document.py   # Pydantic models for documents & chunks
│   │   └── world_model.py # Pydantic models for extraction output
│   ├── services/
│   │   ├── parser.py     # Document parsing (PDF/TXT/MD)
│   │   ├── chunker.py    # Text chunking with overlap
│   │   └── extractor.py  # LLM extraction pipeline
│   ├── api/
│   │   └── routes.py     # API route definitions
│   └── prompts/
│       └── extraction.py # Extraction prompt templates
└── tests/
    └── test_nongfu.py    # Integration test with Nongfu Spring data
```

