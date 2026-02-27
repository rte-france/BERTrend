#  Copyright (c) 2024-2026, RTE (https://www.rte-france.com)
#  See AUTHORS.txt
#  SPDX-License-Identifier: MPL-2.0
#  This file is part of BERTrend.
import hashlib
from datetime import datetime

import requests
from nicegui import run, ui


def _clamp(x, lo=0, hi=255):
    return max(lo, min(hi, x))


def _adjust_color(hex_color, factor):
    """
    Adjust color brightness.
    factor > 1 -> lighten, factor < 1 -> darken.
    """
    hex_color = hex_color.lstrip("#")
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)

    r = _clamp(int(r * factor))
    g = _clamp(int(g * factor))
    b = _clamp(int(b * factor))

    return "#{:02x}{:02x}{:02x}".format(r, g, b)


def get_color(user, model_id, endpoint):
    """
    Generates a stable hex color from (user, model_id).
    - Lighter for 'scrape' endpoints
    - Darker for 'report' endpoints
    """
    seed = f"{user}-{model_id}".encode("utf-8")
    hash_hex = hashlib.md5(seed).hexdigest()
    base_color = f"#{hash_hex[:6]}"

    ep = endpoint.lower()
    if "scrape" in ep:
        # lighten ~30%
        return _adjust_color(base_color, 1.3)
    if "report" in ep:
        # darken ~30%
        return _adjust_color(base_color, 0.7)

    return base_color


def parse_jobs(data):
    parsed = []
    if not isinstance(data, list):
        return []
    for item in data:
        kwargs = item.get("kwargs", {}) or {}
        url = kwargs.get("url", "")
        endpoint = url.split("/")[-1] if url else "N/A"
        json_data = kwargs.get("json_data", {}) or {}
        user = str(json_data.get("user", "system"))
        model_id = json_data.get("model_id", "N/A")
        parsed.append(
            {
                "user": user,
                "model_id": model_id,
                "endpoint": endpoint,
                "time": item.get("next_run_time", ""),
                "job_id": item.get("job_id", "N/A"),
                "color": get_color(user, model_id, endpoint),
            }
        )
    return sorted(parsed, key=lambda x: x["time"])


class JobDashboard:
    def __init__(self):
        self.api_url = "http://dsia.rte-france.com:8882/jobs"
        self.all_jobs = []
        self.filtered_jobs = []

    async def fetch_data(self):
        try:
            response = await run.io_bound(requests.get, self.api_url, timeout=10)
            if response.status_code == 200:
                self.all_jobs = parse_jobs(response.json())
                self.filtered_jobs = self.all_jobs[:]
                ui.notify(f"Fetched {len(self.all_jobs)} jobs", type="positive")
            else:
                ui.notify(f"Error: {response.status_code}", type="negative")
        except Exception as e:
            ui.notify(f"Connection Failed: {e}", type="negative")


@ui.page("/")
async def main_page():
    # Load Highcharts Module
    ui.add_head_html(
        '<script src="https://code.highcharts.com/modules/timeline.js"></script>'
    )
    ui.colors(primary="#0052a1")
    dashboard = JobDashboard()

    def update_display():
        f_users = user_filter.value
        f_models = model_filter.value
        f_urls = url_filter.value

        dashboard.filtered_jobs = [
            j
            for j in dashboard.all_jobs
            if (not f_users or j["user"] in f_users)
            and (not f_models or j["model_id"] in f_models)
            and (not f_urls or j["endpoint"] in f_urls)
        ]
        render_content()

    def reset_filters():
        user_filter.value = []
        model_filter.value = []
        url_filter.value = []
        update_display()

    def render_content():
        # Clear both containers
        chart_container.clear()
        timeline_container.clear()

        if not dashboard.filtered_jobs:
            with chart_container:
                ui.label("No jobs match selected filters.").classes(
                    "mx-auto py-10 text-slate-400"
                )
            return

        # 1. Update Chart Dashboard
        with chart_container:
            chart_options = {
                "chart": {"type": "timeline", "height": 400, "zoomType": "x"},
                "title": {
                    "text": "Job Schedule Timeline",
                    "style": {"fontSize": "14px"},
                },
                "xAxis": {"type": "datetime"},
                "series": [
                    {
                        "data": [
                            {
                                "x": int(
                                    datetime.fromisoformat(j["time"]).timestamp() * 1000
                                ),
                                "name": f"/{j['endpoint']}",
                                "label": f"{j['user']} | {j['model_id']}",
                                "color": j["color"],
                                "dataLabels": {
                                    "style": {
                                        "fontSize": "10px",
                                    }
                                },
                            }
                            for j in dashboard.filtered_jobs
                            if j["time"]
                        ]
                    }
                ],
            }
            ui.highchart(chart_options, extras=["timeline"]).classes("w-full")

        # 2. Update Timeline List Dashboard
        with timeline_container:
            ui.label("Detailed Job Queue").classes(
                "text-sm font-bold text-slate-500 mb-4 uppercase"
            )
            # --- Generate and inject CSS for our custom hex colors ---
            unique_colors = set(job["color"] for job in dashboard.filtered_jobs)
            css_rules = []
            for hex_color in unique_colors:
                # Create a valid CSS identifier from the hex (e.g., #ff0000 -> c_ff0000)
                class_name = f"c_{hex_color.lstrip('#')}"
                css_rules.append(
                    f".text-{class_name} {{ color: {hex_color} !important; }}"
                )
                css_rules.append(
                    f".bg-{class_name} {{ background-color: {hex_color} !important; }}"
                )

            if css_rules:
                ui.add_css("\n".join(css_rules))
            # --------------------------------------------------------------
            with ui.timeline(side="right"):
                for job in dashboard.filtered_jobs:
                    custom_color_name = f"c_{job['color'].lstrip('#')}"
                    ui.timeline_entry(
                        f"Model: {job['model_id']}  |  (Job ID: {job['job_id']})",
                        title=f"/{job['endpoint']} ({job['user']})",
                        subtitle=job["time"],
                        color=custom_color_name,
                    )

    async def refresh_all():
        await dashboard.fetch_data()
        # Refresh options based on new data
        user_filter.set_options(
            sorted(list(set(j["user"] for j in dashboard.all_jobs)))
        )
        model_filter.set_options(
            sorted(list(set(j["model_id"] for j in dashboard.all_jobs)))
        )
        url_filter.set_options(
            sorted(list(set(j["endpoint"] for j in dashboard.all_jobs)))
        )
        update_display()

    # --- UI LAYOUT ---
    with ui.header().classes("items-center justify-between px-4 py-2"):
        with ui.row().classes("items-center"):
            ui.icon("analytics", size="md")
            ui.label("BERTrend Scheduler").classes("text-lg font-bold")

        # Endpoint Config in Header
        with ui.row().classes("items-center bg-white/10 p-1 rounded-md"):
            ui.input(value=dashboard.api_url).bind_value(dashboard, "api_url").props(
                "dark dense borderless"
            ).classes("w-64 px-2 text-sm")
            ui.button(icon="refresh", on_click=refresh_all).props(
                "flat color=white dense"
            )

    # Filter Bar
    with ui.row().classes("w-full items-center p-2 bg-slate-50 border-b gap-3"):
        ui.label("FILTERS:").classes("text-xs font-bold text-slate-500 ml-2")

        url_filter = (
            ui.select(
                [],
                label="URL Type",
                multiple=True,
                with_input=True,
                on_change=update_display,
            )
            .props("dense outlined")
            .classes("w-48 text-xs bg-white")
        )

        user_filter = (
            ui.select(
                [],
                label="User",
                multiple=True,
                with_input=True,
                on_change=update_display,
            )
            .props("dense outlined")
            .classes("w-40 text-xs bg-white")
        )

        model_filter = (
            ui.select(
                [],
                label="Model ID",
                multiple=True,
                with_input=True,
                on_change=update_display,
            )
            .props("dense outlined")
            .classes("w-40 text-xs bg-white")
        )

        ui.button("Reset", on_click=reset_filters, icon="filter_alt_off").props(
            "outline dense"
        ).classes("text-xs")

    # The two Dashboard Containers
    chart_container = ui.column().classes("w-full p-4")
    ui.separator()
    timeline_container = ui.column().classes("w-full p-4")

    # Initial Load
    await refresh_all()


ui.run(title="BERTrend Job Viewer", port=8885)
