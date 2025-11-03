# Newsletter generation

It is possible to generate a newsletter based on the topics found by a BERTopic model using LLM to summarize topics. There are currently two ways of generating newsletter.

## Configuration files

To generate a newsletter, you should first create the following configuration files:

- `newsletter.cfg`: sets parameters for the newsletter and the topic model. You can follow examples in [bertrend_apps/config/newsletters](../bertrend_apps/config/newsletters).
- `feed.cfg`: define on which feed you want to generate the newsletter and at which frequency. You can follow examples in [bertrend_apps/config/feeds](../bertrend_apps/config/feeds).

## One-shot creation

Generate just one newsletter based on a specific dataset. You can use the following command to show help:

```bash
python -m bertrend_apps.newsletters newsletters --help
```

Then create a single newsletter based on your configuration files:
```bash
python -m bertrend_apps.newsletters newsletters newsletter.cfg feed.cfg
```

## Scheduled creation

To automatically send a newsletter at regular intervals, you should define a job that scraps your feed and a job that generated the newsletter.

### New automatic feed installation

Follow this example:

```bash
python -m bertrend_apps.routers schedule-scrapping feed.cfg
```

### New automatic newsletter installation

Follow this example:

```bash
python -m bertrend_apps.newsletters schedule-newsletters newsletter.cfg feed.cfg
```
