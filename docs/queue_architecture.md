# BERTrend Queue Architecture

This document explains the data flow between the Scheduler Service, RabbitMQ Queue, and BERTrend Service.

## Architecture Overview

```mermaid
sequenceDiagram
    participant Scheduler as Scheduler Service
    participant Queue as RabbitMQ Queue
    participant Worker as BERTrend Worker
    participant FastAPI as FastAPI Endpoints
    Note over Scheduler: Job triggers at scheduled time<br/>(cron expression)
    Scheduler ->> Queue: Publish request message<br/>{endpoint, method, json_data}
    Note over Queue: Message stored in<br/>bertrend_requests queue<br/>(with priority)
    Worker ->> Queue: Consume message<br/>(prefetch_count=1)
    Worker ->> Worker: Parse endpoint & json_data
    Worker ->> FastAPI: Call handler function<br/>(e.g., train_new_model, scrape_feed)
    FastAPI ->> FastAPI: Execute business logic
    FastAPI -->> Worker: Return response
    Worker ->> Queue: Publish response<br/>to bertrend_responses queue
    Note over Queue: Response available<br/>for correlation_id lookup
```

## Component Flow Diagram

```mermaid
flowchart TB
    subgraph Scheduler["Scheduler Service"]
        Jobs[(Job Store<br/>APScheduler)]
        Cron[Cron Triggers]
        Jobs --> Cron
    end

    subgraph RabbitMQ["RabbitMQ"]
        ReqQ[["bertrend_requests<br/>(Priority Queue)"]]
        RespQ[["bertrend_responses"]]
        DLQ[["bertrend_failed<br/>(Dead Letter Queue)"]]
    end

    subgraph Worker["BERTrend Worker"]
        Consumer[Message Consumer]
        Router[Endpoint Router]
        Consumer --> Router
    end

    subgraph FastAPI["FastAPI Handlers"]
        SF[/scrape-feed/]
        TM[/train-new-model/]
        RG[/regenerate/]
        GR[/generate-report/]
    end

    Cron -->|" HTTP-like request<br/>{endpoint, json_data} "| ReqQ
    ReqQ -->|" Consume (FIFO + Priority) "| Consumer
    Router --> SF
    Router --> TM
    Router --> RG
    Router --> GR
    SF & TM & RG & GR -->|Response| RespQ
    ReqQ -.->|" Failed after retries "| DLQ
```

## Message Format

### Request Message (Scheduler → Queue)

```json
{
  "endpoint": "/train-new-model",
  "method": "POST",
  "json_data": {
    "user": "celine",
    "model_id": "Capitalisées Municipales 2026"
  }
}
```

### Response Message (Worker → Queue)

```json
{
  "status": "success",
  "endpoint": "/train-new-model",
  "response": {
    "status": "success",
    "message": "Successfully trained model..."
  }
}
```

## Key Features

1. **Priority Queue**: Requests can have priority 1-10 (higher = more urgent)
2. **Sequential Processing**: Worker processes one message at a time (prefetch_count=1)
3. **Dead Letter Queue**: Failed messages after max retries go to `bertrend_failed`
4. **Correlation IDs**: Track request-response pairs for async operations
5. **Durable Queues**: Messages survive RabbitMQ restarts
