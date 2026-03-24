# Data providers

## Description

Grabs articles (news or scientific articles) from the web and store them as jsonlines file.
These collected data can then be used as input of the BERTrend demonstrators.

Several data providers are supported out of the box:

- **Arxiv** — scientific articles from arxiv.org
- **Google News** — news articles via Google News
- **Bing News** — news articles via Bing News API
- **NewsCatcher** — news articles via the NewsCatcher API
- **RSS / ATOM feeds** — any standard RSS or ATOM feed URL
- **Deep Research** — LLM-based deep research provider

## API keys

Some data providers require the creation of an API key to work properly.

This is the case with the Arxiv and NewsCatcher data providers.
The Arxiv data provider uses the Semantic Scholar API to enrich data.
The API can be created for free on their
website (https://www.newscatcherapi.com/, https://www.semanticscholar.org/product/api).

You have then to set the following environment variables:

```bash
export NEWSCATCHER_API_KEY=<your_api_key>
export SEMANTIC_SCHOLAR_API_KEY=<your_api_key>
```

## Usage

```bash
python -m bertrend_apps.routers --help

Usage: python -m routers [OPTIONS] COMMAND [ARGS]...                                                                                           
                                                                                                                                                      
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --install-completion        [bash|zsh|fish|powershell|pwsh]  Install completion for the specified shell. [default: None]                           │
│ --show-completion           [bash|zsh|fish|powershell|pwsh]  Show completion for the specified shell, to copy it or customize the installation.    │
│                                                              [default: None]                                                                       │
│ --help                                                       Show this message and exit.                                                           │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Commands ─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ auto-scrape          Scrape data from Google or Bing news (multiple requests from a configuration file: each line of the file shall be compliant   │
│                      with the following format: <keyword list>;<after_date, format YYYY-MM-DD>;<before_date, format YYYY-MM-DD>)                   │
│ generate-query-file  Generates a query file to be used with the auto-scrape command. This is useful for queries generating many results. This will │
│                      split the broad query into many ones, each one covering an 'interval' (range) in days covered by each atomic request. If you  │
│                      want to cover several keywords, run the command several times with the same output file.                                      │
│ scrape               Scrape data from Google or Bing news (single request).                                                                        │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯

 ```

```bash
python -m bertrend_apps.routers --help auto-scrape --help
                                                                                                                                                      
Usage: python -m routers auto-scrape [OPTIONS] [REQUESTS_FILE]                                                                                 
                                                                                                                                                      
 Scrape data from Google, Bing news or NewsCatcher (multiple requests from a configuration file: each line of the file shall be compliant with the    
 following format: <keyword list>;<after_date, format YYYY-MM-DD>;<before_date, format YYYY-MM-DD>)                                                   
 Parameters ---------- requests_file: str     Text file containing the list of requests to be processed provider: str     News data provider. Current 
 authorized values [google, bing, newscatcher] save_path: str     Path to the output file (jsonl format)                                              
 Returns -------                                                                                                                                      
                                                                                                                                                      
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   requests_file      [REQUESTS_FILE]  path of jsonlines input file containing the expected queries. [default: None]                                │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --max-results        INTEGER  maximum number of results per request [default: 50]                                                                  │
│ --provider           TEXT     source for news [google, bing, newscatcher] [default: google]                                                        │
│ --save-path          TEXT     Path for writing results. [default: None]                                                                            │
│ --help                        Show this message and exit.                                                                                          │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯


```

```bash
python -m bertrend_apps.routers generate-query-file --help

Usage: python -m routers generate-query-file [OPTIONS] [KEYWORDS]                                                                              
                                                                                                                                                      
 Generates a query file to be used with the auto-scrape command. This is useful for queries generating many results. This will split the broad query  
 into many ones, each one covering an 'interval' (range) in days covered by each atomic request. If you want to cover several keywords, run the       
 command several times with the same output file.                                                                                                     
 Parameters ---------- keywords: str     query described as keywords after: str     "from" date, formatted as YYYY-MM-DD before: str     "to" date,   
 formatted as YYYY-MM-DD save_path: str     Path to the output file (jsonl format)                                                                    
 Returns -------                                                                                                                                      
                                                                                                                                                      
╭─ Arguments ────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│   keywords      [KEYWORDS]  keywords for news search engine. [default: None]                                                                       │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --after            TEXT     date after which to consider news [format YYYY-MM-DD] [default: None]                                                  │
│ --before           TEXT     date before which to consider news [format YYYY-MM-DD] [default: None]                                                 │
│ --save-path        TEXT     Path for writing results. File is in jsonl format. [default: None]                                                     │
│ --interval         INTEGER  Range of days of atomic requests [default: 30]                                                                         │
│ --help                                                                                                                                             │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Important note

You may expect a rate of 10-20% of articles not correctly processed because of:

- problem of cookies management
- errors 404, 403

---

## Integrating a new data provider

This section explains how to plug a new data source into BERTrend, covering two scenarios:

1. **Build a new provider** — you want to query a new external source (API, database, custom scraper…) and integrate it
   natively into the BERTrend pipeline.
2. **Bring your own data** — you already have a data source and just want to produce files that BERTrend can consume
   directly, without writing any Python provider class.

---

### Scenario 1 — Build a new provider class

#### Architecture overview

All providers inherit from the abstract base class `DataProvider`
(`bertrend/bertrend_apps/data_provider/data_provider.py`).

```
DataProvider  (ABC)
├── get_articles()       ← abstract, must be implemented
├── _parse_entry()       ← abstract, must be implemented
├── get_articles_batch() ← concrete, calls get_articles() in a loop
├── process_entries()    ← concrete, calls _parse_entry() in parallel (threads)
├── store_articles()     ← concrete, writes JSONL output
├── _get_text()          ← concrete, fetches and cleans article body from a URL
└── parse_date()         ← concrete, normalises any date string
```

You only need to implement the two abstract methods.

#### Step 1 — Create the provider file

Create a new file in `bertrend/bertrend_apps/data_provider/`, e.g. `my_source_provider.py`:

```python
from bertrend.bertrend_apps.data_provider.data_provider import DataProvider
from loguru import logger


class MySourceProvider(DataProvider):
    """Provider for MySource API."""

    def __init__(self):
        super().__init__()          # initialises the article parser (Goose3)
        # initialise your API client here if needed

    # ------------------------------------------------------------------
    # Abstract method 1 — entry point called by the pipeline
    # ------------------------------------------------------------------
    def get_articles(
        self,
        query: str,
        after: str,
        before: str,
        max_results: int,
        language: str = None,
    ) -> list[dict]:
        """Query MySource and return a list of article dicts."""
        # 1. Call your external source to get a list of raw entries
        raw_entries = self._call_my_api(query, after, before, max_results)

        # 2. Parse entries in parallel (uses threads — safe for non-picklable state)
        return self.process_entries(raw_entries, lang_filter=language)

    # ------------------------------------------------------------------
    # Abstract method 2 — converts one raw entry into a BERTrend dict
    # ------------------------------------------------------------------
    def _parse_entry(self, entry: dict) -> dict | None:
        """Convert a single raw API entry into a BERTrend article dict."""
        try:
            url = entry["article_url"]
            published = self.parse_date(entry["pub_date"])   # normalise date

            # If the source only provides a URL (no body), fetch the full text:
            text, title = self._get_text(url)
            if not text:
                return None

            return {
                "title": title,
                "summary": entry.get("excerpt", ""),
                "text": text,           # REQUIRED — main content used by BERTrend
                "timestamp": published, # REQUIRED — format: "YYYY-MM-DD HH:MM:SS"
                "url": url,             # REQUIRED — used to derive the source domain
                "link": url,
            }
        except Exception as e:
            logger.error(f"Error parsing entry: {e}")
            return None

    def _call_my_api(self, query, after, before, max_results):
        """Call the external API and return raw results."""
        # ... your implementation here ...
        return []
```

**Key points:**

- Call `super().__init__()` to initialise the built-in article parser.
- Use `self._get_text(url)` whenever you only have a URL and need to fetch the full article body. It handles blacklisted
  domains, Google News URL decoding, and falls back to a secondary parser automatically.
- Use `self.parse_date(date_string)` to normalise any date format to `"YYYY-MM-DD HH:MM:SS"`.
- Use `self.process_entries(raw_entries, lang_filter=language)` to run `_parse_entry` in parallel over a list of raw
  entries and optionally filter by language.
- Return `None` from `_parse_entry` for entries that cannot be parsed; they are silently dropped.

#### Step 2 — Register the provider

Open `bertrend/bertrend_apps/data_provider/data_provider_utils.py` and add your provider to the `PROVIDERS` dictionary:

```python
from bertrend.bertrend_apps.data_provider.my_source_provider import MySourceProvider

PROVIDERS = {
    "arxiv": ArxivProvider,
    "atom": ATOMFeedProvider,
    "rss": RSSFeedProvider,
    "google": GoogleNewsProvider,
    "bing": BingNewsProvider,
    "newscatcher": NewsCatcherProvider,
    "deep_research": DeepResearchProvider,
    "mysource": MySourceProvider,   # ← add this line
}
```

The key (`"mysource"`) is the string you will pass as the `--provider` argument on the CLI or in feed configuration
files.

#### Step 3 — Use the provider

Once registered, the provider is available everywhere BERTrend accepts a provider name:

```bash
# Single scrape
python -m bertrend_apps.routers scrape \
    --provider mysource \
    --keywords "renewable energy" \
    --after 2024-01-01 --before 2024-03-01 \
    --save-path /tmp/results.jsonl

# Batch scrape from a query file
python -m bertrend_apps.routers auto-scrape \
    --provider mysource \
    --save-path /tmp/results.jsonl \
    requests.txt
```

It is also available via the REST API (`POST /scrape`, `POST /auto-scrape`) and in feed configuration TOML files (see
below).

#### Feed configuration TOML

To use your provider in a scheduled feed, set `provider = "mysource"` in the `[data-feed]` section of your feed TOML
file:

```toml
[data-feed]
id = "my_feed"
provider = "mysource"
query = "renewable energy"
number_of_days = 7
max_results = 100
language = "en"
feed_dir_path = "my_feed"
```

> **Note:** providers that return results in a single batch (like `arxiv`, `rss`, `atom`) should be listed in the
`if provider in (...)` branch inside `scrape_feed_from_config()` in `data_provider_utils.py`. Providers that need
> per-day query splitting (like `google`, `bing`) use the default `auto_scrape` path.

---

### Scenario 2 — Bring your own data (no provider class needed)

If you already have a data pipeline that produces articles, you do not need to write a provider class. You just need to
produce files in the format BERTrend expects.

#### Required output format — JSONL

BERTrend reads data as **JSON Lines** (`.jsonl`): one JSON object per line, each representing one article.

#### Required and optional fields

| Field       | Type  | Required    | Description                                                                               |
|-------------|-------|-------------|-------------------------------------------------------------------------------------------|
| `text`      | `str` | **Yes**     | Main textual content of the article. This is the field BERTrend uses for topic modelling. |
| `timestamp` | `str` | **Yes**     | Publication date/time, formatted as `"YYYY-MM-DD HH:MM:SS"`.                              |
| `url`       | `str` | Recommended | Full URL of the article. Used to derive the `source` domain (e.g. `"www.example.com"`).   |
| `title`     | `str` | Recommended | Article title. Used in topic exploration views.                                           |
| `summary`   | `str` | Optional    | Short excerpt or abstract.                                                                |
| `link`      | `str` | Optional    | Alias for `url` (some providers set both).                                                |

#### Minimal valid example

```jsonl
{"text": "Scientists have discovered a new method to store solar energy...", "timestamp": "2024-03-15 08:30:00", "url": "https://www.example.com/article-1", "title": "New solar storage breakthrough"}
{"text": "The European grid operator announced new interconnection targets...", "timestamp": "2024-03-16 14:00:00", "url": "https://www.example.com/article-2", "title": "EU grid expansion plans"}
```

#### Validation rules applied by BERTrend

When BERTrend loads a JSONL file, it applies the following cleaning steps (see `bertrend/utils/data_loading.py`):

- Rows with a missing or unparseable `timestamp` are **dropped**.
- Rows with a missing or empty `text` are **dropped**.
- Duplicate rows (same `timestamp` + `text`) are **dropped**.
- Duplicate rows with the same `title` are **dropped** (first occurrence kept).
- If `url` is absent, the `source` column is set to `None` (no domain extraction).
- If `title` is absent, it is set to an empty string.

#### Producing JSONL from your own pipeline

You can write JSONL from Python using the `jsonlines` library (already a BERTrend dependency):

```python
import jsonlines
from pathlib import Path

articles = [
    {
        "text": "Full article body here...",
        "timestamp": "2024-03-15 08:30:00",
        "url": "https://www.example.com/article-1",
        "title": "Article title",
        "summary": "Short excerpt.",
    },
    # ...
]

output_path = Path("/path/to/bertrend/data/my_feed/2024-03-15_my_feed.jsonl")
output_path.parent.mkdir(parents=True, exist_ok=True)

with jsonlines.open(output_path, "a") as writer:
    writer.write_all(articles)
```

Or from the command line using `jq` or any tool that produces newline-delimited JSON.

#### Where to place the files

By default, BERTrend looks for feed data under:

```
$BERTREND_BASE_DIR/feeds/<feed_dir_path>/
```

Files should follow the naming convention used by the built-in providers:

```
YYYY-MM-DD_<feed_id>.jsonl
```

For example: `2024-03-15_my_feed.jsonl`.

Once the files are in place, they can be used directly by the BERTrend demonstrators or the `/train` REST endpoint.
