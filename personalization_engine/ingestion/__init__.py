"""
ingestion package — the "front door" for telemetry from V3/V4.

Contains the Feature Store: an append-only log of TelemetryEvents, keyed by
trip. Everything the Learning Loop and Reasoning Engine read comes from here,
so no other layer ever parses raw CAN / scene-graph data itself (§2.1).
"""

from ingestion.feature_store import (
    FeatureStore,
    FeatureStoreError,
    JsonlFeatureStore,
    InMemoryFeatureStore,
)

__all__ = [
    "FeatureStore",
    "FeatureStoreError",
    "JsonlFeatureStore",
    "InMemoryFeatureStore",
]
