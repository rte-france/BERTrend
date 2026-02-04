# Streamlit app: BERTrend Queue Monitor
# Redesigned for clarity, hierarchy, and ease of debugging.

from __future__ import annotations

import base64
import json
import os
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st

# Reuse project configuration for RabbitMQ
from bertrend_apps.services.queue_management.rabbitmq_config import RabbitMQConfig


# ---------- Helpers: Management API client ----------
class RabbitMQAdminClient:
    def __init__(self, cfg: RabbitMQConfig, management_url: str | None = None):
        self.cfg = cfg
        default_url = f"http://{cfg.host}:15672"
        self.base_url = (
            management_url or os.getenv("RABBITMQ_MANAGEMENT_URL") or default_url
        ).rstrip("/")
        self.auth = (cfg.username, cfg.password)
        self._session = requests.Session()
        self._session.auth = self.auth
        self._session.headers.update({"Accept": "application/json"})

    def _vhost_path(self) -> str:
        return quote(self.cfg.virtual_host, safe="")

    def get_queue(self, name: str) -> Dict[str, Any]:
        url = f"{self.base_url}/api/queues/{self._vhost_path()}/{quote(name, safe='')}"
        r = self._session.get(url, timeout=5)
        r.raise_for_status()
        return r.json()

    def peek_messages(self, name: str, count: int = 10) -> List[Dict[str, Any]]:
        url = f"{self.base_url}/api/queues/{self._vhost_path()}/{quote(name, safe='')}/get"
        payload = {
            "count": max(1, min(count, 500)),
            "ackmode": "ack_requeue_true",
            "encoding": "auto",
            "truncate": 50000,
        }
        r = self._session.post(url, json=payload, timeout=10)
        r.raise_for_status()
        return r.json()


# ---------- UI Logic & formatters ----------


def _decode_message_payload(msg: Dict[str, Any]) -> Tuple[str, Any]:
    """Returns (format_type, decoded_object)"""
    body_text = msg.get("payload")
    payload_bytes = None

    # RabbitMQ Management API returns 'payload' (base64 or string)
    # and 'payload_encoding' (string or None)
    encoding = msg.get("payload_encoding")

    if encoding == "base64":
        try:
            payload_bytes = base64.b64decode(body_text)
        except Exception:
            payload_bytes = (
                body_text.encode("utf-8", errors="replace")
                if isinstance(body_text, str)
                else body_text
            )
    else:
        if isinstance(body_text, str):
            # Try base64 anyway as a fallback if not specified but looks like it
            try:
                payload_bytes = base64.b64decode(body_text)
            except Exception:
                payload_bytes = body_text.encode("utf-8", errors="replace")
        elif isinstance(body_text, (bytes, bytearray)):
            payload_bytes = bytes(body_text)

    if payload_bytes is None:
        return ("Text", body_text)

    # Try JSON (Preferred)
    try:
        obj = json.loads(payload_bytes.decode("utf-8"))
        return ("JSON", obj)
    except Exception:
        pass

    # Try plain text
    try:
        return ("Text", payload_bytes.decode("utf-8"))
    except Exception:
        pass

    # Fallback to display as hex/repr if everything else fails
    return (
        "Binary",
        str(payload_bytes[:200]) + ("..." if len(payload_bytes) > 200 else ""),
    )


def render_queue_config_grid(qinfo: Dict[str, Any]):
    """Renders technical queue_management config in a compact way."""
    params = {
        "Durable": str(qinfo.get("durable", False)),
        "Auto-Delete": str(qinfo.get("auto_delete", False)),
        "Exclusive": str(qinfo.get("exclusive", False)),
        "Policy": qinfo.get("policy", "None"),
        "Consumers": qinfo.get("consumers", 0),
        "State": qinfo.get("state", "unknown"),
    }

    # CSS grid-like display
    cols = st.columns(len(params))
    for col, (k, v) in zip(cols, params.items()):
        col.markdown(f"**{k}**")
        col.caption(v)


# ---------- Main App ----------

st.set_page_config(
    page_title="BERTrend Queue Monitor",
    page_icon="📬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for tighter UI
st.markdown(
    """
    <style>
        .stMetric {
            background-color: #f0f2f6;
            padding: 10px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
        }
        div[data-testid="stExpander"] details summary p {
            font-family: 'Source Code Pro', monospace;
            font-size: 0.9em; 
            font-weight: 600;
        }
    </style>
""",
    unsafe_allow_html=True,
)

# --- Sidebar Configuration ---
cfg = RabbitMQConfig()
admin_url_default = f"http://{cfg.host}:15672"

with st.sidebar:
    st.title("⚙️ Settings")

    with st.expander("Connection Details", expanded=False):
        mgmt_url = st.text_input(
            "Mgmt URL", value=os.getenv("RABBITMQ_MANAGEMENT_URL", admin_url_default)
        )
        vhost = st.text_input("VHost", value=cfg.virtual_host)
        username = st.text_input("User", value=cfg.username)
        password = st.text_input("Pass", value=cfg.password, type="password")

    st.subheader("Monitoring Targets")
    default_queues = [cfg.request_queue, cfg.response_queue, "bertrend_failed"]
    # Handle duplicates if defaults are same
    default_queues = list(set(default_queues))

    selected_queues = st.multiselect(
        "Queues to Watch",
        options=default_queues
        + ["amq.direct", "celery"],  # Add common extras if needed
        default=default_queues,
    )

    st.divider()

    # Auto-refresh Logic
    c1, c2 = st.columns([1, 2])
    auto_refresh = c1.toggle("Live", value=False)
    refresh_rate = c2.select_slider(
        "Refresh (s)", options=[5, 10, 30, 60, 120], value=30
    )

    if auto_refresh:
        st.caption(f"Refreshing every {refresh_rate}s...")

# Initialize Client
client_cfg = RabbitMQConfig(
    host=cfg.host,
    port=cfg.port,
    username=username,
    password=password,
    virtual_host=vhost,
)
client = RabbitMQAdminClient(client_cfg, management_url=mgmt_url)

# --- Data Fetching ---
queue_data = {}
global_stats = {"total": 0, "ready": 0, "unacked": 0, "consumers": 0}
error_msg = None

try:
    for q in selected_queues:
        info = client.get_queue(q)
        queue_data[q] = info

        # Aggregation
        global_stats["total"] += info.get("messages", 0)
        global_stats["ready"] += info.get("messages_ready", 0)
        global_stats["unacked"] += info.get("messages_unacknowledged", 0)
        global_stats["consumers"] += info.get("consumers", 0)

except Exception as e:
    error_msg = str(e)

# --- Main Dashboard ---

st.title("📬 BERTrend Queue Monitor")
if error_msg:
    st.error(f"Connection Error: {error_msg}")
    st.stop()

# 1. Global Overview Row
st.markdown("### 📊 System Overview")
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric(
    "Total Backlog", global_stats["total"], help="Total messages in all selected queues"
)
kpi2.metric("Ready to Process", global_stats["ready"], delta_color="normal")
kpi3.metric(
    "Unacknowledged",
    global_stats["unacked"],
    delta_color="inverse",
    help="Messages currently being processed but not acked",
)
kpi4.metric("Active Consumers", global_stats["consumers"])

st.divider()

# 2. Queue Tabs
if not selected_queues:
    st.info("Select queues in the sidebar to begin monitoring.")
else:
    tabs = st.tabs([f"📦 {q}" for q in selected_queues])

    for tab, qname in zip(tabs, selected_queues):
        info = queue_data.get(qname, {})
        with tab:
            # A. Header Stats for this specific queue
            rate = ((info.get("message_stats") or {}).get("publish_details") or {}).get(
                "rate", 0.0
            )

            c1, c2, c3, c4 = st.columns([1, 1, 1, 3])
            c1.markdown(f"**Ready:** `{info.get('messages_ready', 0)}`")
            c2.markdown(f"**Unacked:** `{info.get('messages_unacknowledged', 0)}`")
            c3.markdown(f"**Rate:** `{rate:.2f}/s`")
            with c4:
                # Config Grid
                render_queue_config_grid(info)

            st.divider()

            # B. Message Peeking Interface
            c_head, c_peek_n, c_sort = st.columns([3, 1, 1])
            c_head.subheader("Message Browser")

            peek_n = c_peek_n.number_input(
                "Count",
                min_value=1,
                max_value=500,
                value=20,
                key=f"num_{qname}",
            )

            sort_order = c_sort.selectbox(
                "Sort",
                options=["Newest First", "Oldest First"],
                key=f"sort_{qname}",
            )

            # Fetch messages
            msgs = []
            try:
                # We might want to fetch more if we're filtering locally
                # but for simplicity let's stick to peek_n for now.
                # If we fetch more, we can provide better filtering.
                fetch_count = (
                    peek_n
                    if not (
                        st.session_state.get(f"f_end_{qname}")
                        or st.session_state.get(f"f_user_{qname}")
                        or st.session_state.get(f"f_model_{qname}")
                    )
                    else 200
                )
                msgs = client.peek_messages(qname, count=fetch_count)
            except Exception as e:
                st.error(f"Failed to peek: {e}")

            if not msgs:
                st.info("Queue is empty or messages are not available.", icon="ℹ️")
            else:
                # Decode all first for filtering/sorting/collecting filter values
                decoded_msgs = []
                unique_endpoints = set()
                unique_users = set()
                unique_models = set()

                for m in msgs:
                    fmt, obj = _decode_message_payload(m)

                    # Extract fields for filtering
                    endpoint = ""
                    user = ""
                    model_id = ""

                    if isinstance(obj, dict):
                        endpoint = str(obj.get("endpoint", ""))
                        # Check json_data if present
                        json_data = obj.get("json_data", {})
                        if isinstance(json_data, dict):
                            user = str(json_data.get("user", ""))
                            model_id = str(json_data.get("model_id", ""))

                        # Fallback to top level if not in json_data
                        user = user or str(obj.get("user", ""))
                        model_id = model_id or str(obj.get("model_id", ""))

                    if endpoint:
                        unique_endpoints.add(endpoint)
                    if user:
                        unique_users.add(user)
                    if model_id:
                        unique_models.add(model_id)

                    decoded_msgs.append(
                        {
                            "raw": m,
                            "fmt": fmt,
                            "obj": obj,
                            "endpoint": endpoint,
                            "user": user,
                            "model_id": model_id,
                            "timestamp": m.get("properties", {}).get("timestamp", 0),
                        }
                    )

                # Filtering inputs
                with st.expander("🔍 Filters", expanded=False):
                    fc1, fc2, fc3 = st.columns(3)

                    # Options for selectboxes (sorted for better UX)
                    opt_endpoints = ["All"] + sorted(list(unique_endpoints))
                    opt_users = ["All"] + sorted(list(unique_users))
                    opt_models = ["All"] + sorted(list(unique_models))

                    f_endpoint = fc1.selectbox(
                        "Endpoint", options=opt_endpoints, key=f"f_end_{qname}"
                    )
                    f_user = fc2.selectbox(
                        "User", options=opt_users, key=f"f_user_{qname}"
                    )
                    f_model = fc3.selectbox(
                        "Model ID", options=opt_models, key=f"f_model_{qname}"
                    )

                # Apply Filtering
                if f_endpoint and f_endpoint != "All":
                    decoded_msgs = [
                        d for d in decoded_msgs if d["endpoint"] == f_endpoint
                    ]
                if f_user and f_user != "All":
                    decoded_msgs = [d for d in decoded_msgs if d["user"] == f_user]
                if f_model and f_model != "All":
                    decoded_msgs = [d for d in decoded_msgs if d["model_id"] == f_model]

                # Apply Sorting
                if sort_order == "Newest First":
                    # If no timestamp, keep original order (which is usually oldest first in RabbitMQ)
                    # so we reverse it.
                    decoded_msgs.reverse()

                # Limit to peek_n
                display_msgs = decoded_msgs[:peek_n]

                if not display_msgs:
                    st.warning("No messages match the current filters.")

                # Convert list of msgs to a more displayable format
                for idx, d in enumerate(display_msgs):
                    m = d["raw"]
                    fmt = d["fmt"]
                    obj = d["obj"]

                    props = m.get("properties", {})
                    corr_id = props.get("correlation_id", "No-ID")
                    msg_size = (
                        m.get("payload_bytes", 0)
                        if isinstance(m.get("payload_bytes"), int)
                        else len(str(m.get("payload")))
                    )

                    # Collapsible Message Row
                    # Show endpoint/user in label if available
                    info_tags = []
                    if d["endpoint"]:
                        info_tags.append(d["endpoint"])
                    if d["user"]:
                        info_tags.append(f"User: {d['user']}")

                    info_str = f" | {' | '.join(info_tags)}" if info_tags else ""
                    label = f"#{idx + 1} | {fmt} | CID: {corr_id}{info_str} | {msg_size} bytes"

                    with st.expander(label, expanded=False):
                        mc1, mc2 = st.columns([3, 1])

                        # Payload column
                        with mc1:
                            st.caption(f"Payload Content ({fmt})")
                            if isinstance(obj, (dict, list)):
                                st.json(
                                    obj, expanded=(idx == 0)
                                )  # Expand only the first one by default
                            else:
                                st.code(str(obj), language="text")

                            with st.expander("Raw Payload", expanded=False):
                                st.code(str(m.get("payload")), language="text")

                        # Metadata column
                        with mc2:
                            st.caption("Properties")
                            meta_df = pd.DataFrame(
                                [
                                    {
                                        "Key": "Exchange",
                                        "Val": str(m.get("exchange") or ""),
                                    },
                                    {
                                        "Key": "Routing Key",
                                        "Val": str(m.get("routing_key") or ""),
                                    },
                                    {
                                        "Key": "Redelivered",
                                        "Val": str(m.get("redelivered")),
                                    },
                                    {
                                        "Key": "Priority",
                                        "Val": str(props.get("priority") or ""),
                                    },
                                    {
                                        "Key": "Timestamp",
                                        "Val": str(props.get("timestamp") or ""),
                                    },
                                    {
                                        "Key": "App ID",
                                        "Val": str(props.get("app_id") or ""),
                                    },
                                ]
                            ).dropna()
                            st.dataframe(meta_df, hide_index=True, width="stretch")

# Auto-refresh handler at the very end
if auto_refresh:
    time.sleep(refresh_rate)
    st.rerun()
