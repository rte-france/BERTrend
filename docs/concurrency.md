# Concurrency in BERTrend

This document describes where and why multiprocessing and multithreading are used across the BERTrend codebase.

## Overview

BERTrend uses concurrency in several areas to improve performance for I/O-bound and CPU-bound tasks. The table below
summarizes each usage:

| Location                           | Mechanism                           | Purpose                                     |
|------------------------------------|-------------------------------------|---------------------------------------------|
| `process_new_data.py`              | `ThreadPoolExecutor`                | Parallel LLM-based signal analysis          |
| `feeds_data.py`                    | `ThreadPoolExecutor`                | Parallel file loading for feed data         |
| `data_provider.py`                 | `joblib.Parallel` (threads)         | Parallel parsing of feed entries            |
| `models_info.py`                   | `multiprocessing.Process`           | Non-blocking model regeneration from the UI |
| `scheduling/routers/scheduling.py` | `ProcessPoolExecutor` (APScheduler) | Concurrent scheduled job execution          |

---

## Detailed Descriptions

### 1. Parallel Signal Analysis — `process_new_data.py`

**File:** `bertrend/bertrend_apps/prospective_demo/process_new_data.py`

**Mechanism:** `concurrent.futures.ThreadPoolExecutor` with `as_completed`

**Purpose:** After BERTrend classifies topics into weak signals, strong signals, and noise, each topic requires an
LLM-based interpretation via `analyze_signal()`. This involves network calls to an LLM API, making it I/O-bound. A
`ThreadPoolExecutor` processes multiple topics concurrently, significantly reducing the total time needed to generate
interpretations for all topics.

**How it works:**

- Each topic is submitted as a separate task to the thread pool.
- Results are collected as they complete (`as_completed`), and failures are logged without stopping other tasks.
- Results are sorted by topic number after collection to maintain a consistent output order.

### 2. Parallel Feed Data Loading — `feeds_data.py`

**File:** `bertrend/bertrend_apps/prospective_demo/feeds_data.py`

**Mechanism:** `concurrent.futures.ThreadPoolExecutor` with `executor.map`

**Purpose:** When displaying data status in the Streamlit UI, all data files for a given feed must be loaded and
concatenated. Since file I/O is the bottleneck, a `ThreadPoolExecutor` loads multiple files in parallel. The number of
workers is set to `os.cpu_count() - 1` (minimum 1) to maximize throughput without starving the system.

### 3. Parallel Entry Parsing — `data_provider.py`

**File:** `bertrend/bertrend_apps/data_provider/data_provider.py`

**Mechanism:** `joblib.Parallel` with `prefer="threads"`

**Purpose:** When processing entries from data providers (RSS, Atom, Google News, etc.), each entry requires network
requests to fetch and parse article content (`_parse_entry` → `_get_text`). This is I/O-bound work. `joblib.Parallel`
with `n_jobs=-1` uses all available CPUs to process entries concurrently.

**Why threads instead of processes:** A thread-based backend is explicitly chosen because the `DataProvider` instance (
`self`) holds non-picklable state (e.g., Goose3 parser, network clients). Process-based backends (loky/multiprocessing)
would require serializing `self`, which would fail. Threads share the same memory space and can safely access this
state.

### 4. Non-blocking Model Regeneration — `models_info.py`

**File:** `bertrend/bertrend_apps/prospective_demo/models_info.py`

**Mechanism:** `multiprocessing.Process` with `spawn` start method

**Purpose:** When a user triggers model regeneration from the Streamlit UI, the operation is long-running (training
BERTopic models, computing embeddings, etc.). Launching it in a separate **process** (not thread) avoids blocking the
Streamlit event loop and keeps the UI responsive.

**Why `spawn` start method:** The `spawn` method is required because the child process uses CUDA (GPU). The default
`fork` method on Linux would duplicate the parent's CUDA context, leading to errors. `spawn` starts a fresh Python
interpreter, ensuring clean CUDA initialization. The target GPU is selected via `CUDA_VISIBLE_DEVICES` before spawning.

### 5. Scheduled Job Execution — `scheduling/routers/scheduling.py`

**File:** `bertrend/services/scheduling/routers/scheduling.py`

**Mechanism:** APScheduler `BackgroundScheduler` with `ProcessPoolExecutor`

**Purpose:** The scheduling service manages recurring jobs (e.g., periodic data gathering, model training) via a FastAPI
endpoint. APScheduler's `BackgroundScheduler` runs in the background and dispatches jobs to a `ProcessPoolExecutor` (
configured with up to 100 workers). This allows multiple scheduled jobs to execute truly in parallel, isolated in
separate processes.

**Why processes instead of threads:** Scheduled jobs (data gathering, model training) can be CPU-intensive and
long-running. Process-based execution provides true parallelism (bypassing the GIL) and process isolation, so a crash in
one job does not affect others. Job functions must be defined at module level to be picklable.

**Configuration:**

- `MAX_WORKERS = 100`: Supports many simultaneous data-gathering jobs.
- `MAX_INSTANCES = 1`: Prevents the same job from running concurrently with itself.
- Jobs are persisted in SQLite via `SQLAlchemyJobStore`, surviving service restarts.
