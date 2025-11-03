#  Copyright (c) 2024, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.

import pytest
from pathlib import Path
from pydantic import ValidationError

from bertrend_apps.services.models.bertrend_app_models import (
    TrainNewModelRequest,
    RegenerateRequest,
    StatusResponse,
)
from bertrend_apps.services.models.newsletters_models import (
    NewsletterRequest,
    NewsletterResponse,
    ScheduleNewsletterRequest,
)
from bertrend_apps.services.models.data_provider_models import (
    ScrapeRequest,
    ScrapeResponse,
    AutoScrapeRequest,
    GenerateQueryFileRequest,
)


class TestBertrendAppModels:
    """Tests for bertrend_app_models.py"""

    def test_train_new_model_request_valid(self):
        req = TrainNewModelRequest(user="test_user", model_id="model_123")
        assert req.user == "test_user"
        assert req.model_id == "model_123"

    def test_train_new_model_request_missing_fields(self):
        with pytest.raises(ValidationError):
            TrainNewModelRequest(user="test_user")
        with pytest.raises(ValidationError):
            TrainNewModelRequest(model_id="model_123")

    def test_train_new_model_response_valid(self):
        resp = StatusResponse(status="success", message="Model trained")
        assert resp.status == "success"
        assert resp.message == "Model trained"

    def test_regenerate_request_valid(self):
        req = RegenerateRequest(user="test_user", model_id="model_123")
        assert req.user == "test_user"
        assert req.model_id == "model_123"
        assert req.with_analysis is True  # default value
        assert req.since is None  # default value

    def test_regenerate_request_with_optional_fields(self):
        req = RegenerateRequest(
            user="test_user",
            model_id="model_123",
            with_analysis=False,
            since="2025-01-01",
        )
        assert req.with_analysis is False
        assert req.since == "2025-01-01"

    def test_regenerate_response_valid(self):
        resp = StatusResponse(status="success", message="Regenerated")
        assert resp.status == "success"
        assert resp.message == "Regenerated"


class TestNewslettersModels:
    """Tests for newsletters_models.py"""

    def test_newsletter_request_valid(self, tmp_path):
        newsletter_path = tmp_path / "newsletter.toml"
        feed_path = tmp_path / "feed.toml"
        req = NewsletterRequest(
            newsletter_toml_path=newsletter_path, data_feed_toml_path=feed_path
        )
        assert req.newsletter_toml_path == newsletter_path
        assert req.data_feed_toml_path == feed_path

    def test_newsletter_request_missing_fields(self):
        with pytest.raises(ValidationError):
            NewsletterRequest(newsletter_toml_path=Path("test.toml"))

    def test_newsletter_response_valid(self, tmp_path):
        output_path = tmp_path / "output.html"
        resp = NewsletterResponse(output_path=output_path, status="success")
        assert resp.output_path == output_path
        assert resp.status == "success"

    def test_schedule_newsletter_request_valid(self, tmp_path):
        newsletter_path = tmp_path / "newsletter.toml"
        feed_path = tmp_path / "feed.toml"
        req = ScheduleNewsletterRequest(
            newsletter_toml_cfg_path=newsletter_path,
            data_feed_toml_cfg_path=feed_path,
        )
        assert req.newsletter_toml_cfg_path == newsletter_path
        assert req.data_feed_toml_cfg_path == feed_path
        assert req.cuda_devices is None  # default value

    def test_schedule_newsletter_request_with_cuda(self, tmp_path):
        newsletter_path = tmp_path / "newsletter.toml"
        feed_path = tmp_path / "feed.toml"
        req = ScheduleNewsletterRequest(
            newsletter_toml_cfg_path=newsletter_path,
            data_feed_toml_cfg_path=feed_path,
            cuda_devices="0,1",
        )
        assert req.cuda_devices == "0,1"


class TestDataProviderModels:
    """Tests for data_provider_models.py"""

    def test_scrape_request_minimal(self):
        req = ScrapeRequest(keywords="ai")
        assert req.keywords == "ai"
        assert req.provider == "google"  # default
        assert req.max_results == 50  # default
        assert req.after is None
        assert req.before is None
        assert req.save_path is None
        assert req.language is None

    def test_scrape_request_full(self, tmp_path):
        save_path = tmp_path / "output.jsonl"
        req = ScrapeRequest(
            keywords="machine learning",
            provider="arxiv",
            after="2025-01-01",
            before="2025-01-31",
            max_results=100,
            save_path=save_path,
            language="en",
        )
        assert req.keywords == "machine learning"
        assert req.provider == "arxiv"
        assert req.after == "2025-01-01"
        assert req.before == "2025-01-31"
        assert req.max_results == 100
        assert req.save_path == save_path
        assert req.language == "en"

    def test_scrape_response_valid(self, tmp_path):
        stored_path = tmp_path / "results.jsonl"
        resp = ScrapeResponse(stored_path=stored_path, article_count=42)
        assert resp.stored_path == stored_path
        assert resp.article_count == 42

    def test_scrape_response_no_path(self):
        resp = ScrapeResponse(stored_path=None, article_count=5)
        assert resp.stored_path is None
        assert resp.article_count == 5

    def test_auto_scrape_request_minimal(self, tmp_path):
        requests_file = tmp_path / "requests.txt"
        req = AutoScrapeRequest(requests_file=str(requests_file))
        assert req.requests_file == str(requests_file)
        assert req.max_results == 50  # default
        assert req.provider == "google"  # default
        assert req.evaluate_articles_quality is False  # default
        assert req.minimum_quality_level == "AVERAGE"  # default

    def test_auto_scrape_request_full(self, tmp_path):
        requests_file = tmp_path / "requests.txt"
        save_path = tmp_path / "output.jsonl"
        req = AutoScrapeRequest(
            requests_file=str(requests_file),
            max_results=100,
            provider="bing",
            save_path=save_path,
            language="fr",
            evaluate_articles_quality=True,
            minimum_quality_level="GOOD",
        )
        assert req.max_results == 100
        assert req.provider == "bing"
        assert req.save_path == save_path
        assert req.language == "fr"
        assert req.evaluate_articles_quality is True
        assert req.minimum_quality_level == "GOOD"

    def test_generate_query_file_request_valid(self, tmp_path):
        save_path = tmp_path / "queries.txt"
        req = GenerateQueryFileRequest(
            keywords="climate",
            after="2025-01-01",
            before="2025-01-31",
            save_path=save_path,
        )
        assert req.keywords == "climate"
        assert req.after == "2025-01-01"
        assert req.before == "2025-01-31"
        assert req.save_path == save_path
        assert req.interval == 30  # default

    def test_generate_query_file_request_custom_interval(self, tmp_path):
        save_path = tmp_path / "queries.txt"
        req = GenerateQueryFileRequest(
            keywords="climate",
            after="2025-01-01",
            before="2025-01-31",
            save_path=save_path,
            interval=7,
        )
        assert req.interval == 7
