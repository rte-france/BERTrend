#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
from pathlib import Path

import pytest

from bertrend.bertrend_apps.common import portable_paths
from bertrend.bertrend_apps.common.portable_paths import (
    remap_legacy_path,
    resolve_portable_path,
    to_portable_path,
)

USER_FEED = Path("config/users/celine/datacenter_feed.toml")
PACKAGED_FEED = Path("bertrend_apps/config/feeds/rte_feed.toml")


@pytest.fixture
def anchors(tmp_path, monkeypatch):
    """A BERTREND_BASE_DIR and a package dir, each holding one config file."""
    base, package = tmp_path / "base", tmp_path / "site-packages" / "bertrend"
    for f in (base / USER_FEED, package / PACKAGED_FEED):
        f.parent.mkdir(parents=True)
        f.write_text("")
    monkeypatch.setattr(portable_paths, "BASE_PATH", base)
    monkeypatch.setattr(portable_paths, "PACKAGE_PATH", package)
    return base, package


class TestToPortablePath:
    def test_under_base_dir_is_relative_to_it(self, anchors):
        base, _ = anchors
        assert to_portable_path(base / USER_FEED) == USER_FEED.as_posix()

    def test_under_package_is_relative_to_it(self, anchors):
        _, package = anchors
        assert to_portable_path(package / PACKAGED_FEED) == PACKAGED_FEED.as_posix()

    def test_outside_anchors_stays_absolute(self, anchors, tmp_path):
        other = tmp_path / "elsewhere" / "feed.toml"
        assert to_portable_path(other) == other.as_posix()

    def test_relative_input_is_taken_from_the_current_dir(self, anchors, monkeypatch):
        base, _ = anchors
        monkeypatch.chdir(base / "config")
        assert (
            to_portable_path("users/celine/datacenter_feed.toml")
            == USER_FEED.as_posix()
        )


class TestResolvePortablePath:
    def test_relative_to_base_dir(self, anchors):
        base, _ = anchors
        assert resolve_portable_path(USER_FEED.as_posix()) == base / USER_FEED

    def test_relative_to_package(self, anchors):
        _, package = anchors
        assert resolve_portable_path(PACKAGED_FEED) == package / PACKAGED_FEED

    def test_relative_not_under_anchors_is_returned_as_given(self, anchors):
        assert resolve_portable_path("feeds/missing.toml") == Path("feeds/missing.toml")

    def test_existing_absolute_path_is_kept(self, anchors, tmp_path):
        other = tmp_path / "other.toml"
        other.write_text("")
        assert resolve_portable_path(other) == other

    def test_legacy_host_base_dir_path_is_remapped(self, anchors, tmp_path):
        base, _ = anchors
        legacy = tmp_path / "old-host" / "nlp" / "bertrend" / USER_FEED
        assert resolve_portable_path(legacy) == base / USER_FEED

    def test_legacy_source_checkout_path_is_remapped(self, anchors, tmp_path):
        _, package = anchors
        legacy = tmp_path / "old-host" / "dev" / "BERTrend" / "bertrend" / PACKAGED_FEED
        assert resolve_portable_path(legacy) == package / PACKAGED_FEED

    def test_unknown_absolute_path_is_returned_as_given(self, anchors, tmp_path):
        unknown = tmp_path / "old-host" / "x" / "y.toml"
        assert resolve_portable_path(unknown) == unknown


class TestRemapLegacyPath:
    def test_bare_file_name_is_not_matched(self, anchors, tmp_path):
        base, _ = anchors
        (base / "feed.toml").write_text("")
        assert remap_legacy_path(tmp_path / "old-host" / "feed.toml") is None

    def test_longest_suffix_wins(self, anchors, tmp_path):
        base, _ = anchors
        # Same trailing "<user>/<file>" in two places: the longer match is chosen
        decoy = base / "users" / "celine" / "datacenter_feed.toml"
        decoy.parent.mkdir(parents=True)
        decoy.write_text("")
        legacy = tmp_path / "old-host" / USER_FEED
        assert remap_legacy_path(legacy) == base / USER_FEED
