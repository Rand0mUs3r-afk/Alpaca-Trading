"""Tests for FinMem-style memory maintenance: time/performance decay,
duplicate pruning, and per-collection size caps.

Adaptation of FinMem (arXiv:2311.13743) to a single-layer ChromaDB store:
instead of discrete shallow/intermediate/deep layers, each entry's decay
stability interpolates continuously with its importance — important lessons
(large realized moves) decay like FinMem's deep layer (Q=365d), trivial
ones like the shallow layer (Q=14d). Entries purge when recency < 0.05,
near-duplicates (cosine similarity above threshold) collapse to the
highest-scoring copy, and a size cap evicts the lowest-scoring entries.
"""

import math
import unittest
import uuid
from datetime import date, timedelta

from tradingagents.agents.utils.memory import FinancialSituationMemory
from tradingagents.agents.utils.memory_maintenance import (
    MemoryMaintenanceConfig,
    entry_importance,
    entry_recency,
    maintain_memory,
)


def _fresh_memory():
    memory = FinancialSituationMemory(f"maint_test_{uuid.uuid4().hex[:8]}")
    memory.embeddings_enabled = True
    return memory


def _add(memory, text, embedding, **metadata):
    memory.get_embedding = lambda _t: embedding
    memory.add_situations([(text, f"advice for {text}")], extra_metadata=metadata or None)


def _days_ago(days):
    return (date.today() - timedelta(days=days)).isoformat()


class ScoringTests(unittest.TestCase):
    def test_recency_is_ebbinghaus_decay(self):
        # importance 0 -> shallow stability (14 days).
        meta = {"trade_date": _days_ago(14)}
        self.assertAlmostEqual(
            entry_recency(meta, importance=0.0), math.exp(-1.0), places=6
        )

    def test_high_importance_decays_slower(self):
        meta = {"trade_date": _days_ago(180)}
        shallow = entry_recency(meta, importance=0.0)
        deep = entry_recency(meta, importance=1.0)
        self.assertLess(shallow, 0.05)  # 180d >> Q_shallow=14d: expired
        self.assertGreater(deep, 0.5)  # 180d < Q_deep=365d: well retained

    def test_undated_entry_gets_neutral_recency(self):
        self.assertEqual(entry_recency({}, importance=0.0), 0.5)

    def test_importance_scales_with_realized_return(self):
        base = entry_importance({})
        small = entry_importance({"decision_return": 0.005})
        large = entry_importance({"decision_return": -0.25})
        self.assertGreater(small, base - 1e-9)
        self.assertGreater(large, small)
        self.assertLessEqual(large, 1.0)

    def test_created_at_used_when_no_trade_date(self):
        meta = {"created_at": _days_ago(14)}
        self.assertAlmostEqual(
            entry_recency(meta, importance=0.0), math.exp(-1.0), places=6
        )


class CreatedAtStampTests(unittest.TestCase):
    def test_add_situations_stamps_created_at(self):
        memory = _fresh_memory()
        _add(memory, "sit", [0.1] * 8)
        meta = memory.situation_collection.get(include=["metadatas"])["metadatas"][0]
        self.assertEqual(meta["created_at"], date.today().isoformat())

    def test_explicit_created_at_not_overwritten(self):
        memory = _fresh_memory()
        _add(memory, "sit", [0.1] * 8, created_at="2020-01-01")
        meta = memory.situation_collection.get(include=["metadatas"])["metadatas"][0]
        self.assertEqual(meta["created_at"], "2020-01-01")


class MaintainMemoryTests(unittest.TestCase):
    def test_expired_entries_are_purged(self):
        memory = _fresh_memory()
        # Low importance + very old: recency far below 0.05.
        _add(memory, "ancient", [0.1] * 8, trade_date=_days_ago(400))
        _add(memory, "recent", [0.9] * 8, trade_date=_days_ago(2))
        summary = maintain_memory(memory)
        self.assertEqual(summary["purged_expired"], 1)
        self.assertEqual(memory.situation_collection.count(), 1)
        kept = memory.situation_collection.get(include=["documents"])["documents"]
        self.assertEqual(kept, ["recent"])

    def test_important_old_entries_survive(self):
        memory = _fresh_memory()
        # Same age, but a large realized move -> deep-layer stability.
        _add(
            memory,
            "big lesson",
            [0.1] * 8,
            trade_date=_days_ago(180),
            decision_return=0.30,
        )
        summary = maintain_memory(memory)
        self.assertEqual(summary["purged_expired"], 0)
        self.assertEqual(memory.situation_collection.count(), 1)

    def test_near_duplicates_collapse_to_one(self):
        memory = _fresh_memory()
        _add(memory, "dup one", [0.5] * 8, trade_date=_days_ago(5))
        _add(memory, "dup two", [0.5] * 8, trade_date=_days_ago(1))
        _add(memory, "distinct", [0.9, -0.4, 0.2, 0.7, -0.1, 0.3, -0.8, 0.6])
        summary = maintain_memory(memory)
        self.assertEqual(summary["purged_duplicates"], 1)
        docs = memory.situation_collection.get(include=["documents"])["documents"]
        self.assertIn("distinct", docs)
        # The newer duplicate wins on recency.
        self.assertIn("dup two", docs)
        self.assertNotIn("dup one", docs)

    def test_size_cap_evicts_lowest_scores(self):
        memory = _fresh_memory()
        embeddings = [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        _add(memory, "old weak", embeddings[0], trade_date=_days_ago(10))
        _add(memory, "fresh", embeddings[1], trade_date=_days_ago(1))
        _add(
            memory,
            "important",
            embeddings[2],
            trade_date=_days_ago(5),
            decision_return=0.2,
        )
        config = MemoryMaintenanceConfig(max_entries=2)
        summary = maintain_memory(memory, config)
        self.assertEqual(summary["purged_over_cap"], 1)
        docs = memory.situation_collection.get(include=["documents"])["documents"]
        self.assertEqual(memory.situation_collection.count(), 2)
        self.assertNotIn("old weak", docs)

    def test_maintenance_is_idempotent_when_healthy(self):
        memory = _fresh_memory()
        _add(memory, "a", [0.1] * 8, trade_date=_days_ago(1))
        _add(memory, "b", [0.9, -0.4, 0.2, 0.7, -0.1, 0.3, -0.8, 0.6])
        first = maintain_memory(memory)
        second = maintain_memory(memory)
        self.assertEqual(second["purged_expired"], 0)
        self.assertEqual(second["purged_duplicates"], 0)
        self.assertEqual(second["purged_over_cap"], 0)
        self.assertEqual(second["kept"], first["kept"])

    def test_empty_memory_is_noop(self):
        memory = _fresh_memory()
        summary = maintain_memory(memory)
        self.assertEqual(summary["kept"], 0)


class ConfigWiringTests(unittest.TestCase):
    def test_default_config_exposes_maintenance_keys(self):
        from tradingagents.default_config import DEFAULT_CONFIG

        self.assertIn("memory_maintenance_enabled", DEFAULT_CONFIG)
        self.assertIn("memory_max_entries_per_collection", DEFAULT_CONFIG)

    def test_config_from_dict_reads_project_config(self):
        config = MemoryMaintenanceConfig.from_config(
            {
                "memory_max_entries_per_collection": 42,
                "memory_duplicate_similarity_threshold": 0.9,
            }
        )
        self.assertEqual(config.max_entries, 42)
        self.assertEqual(config.duplicate_similarity, 0.9)


if __name__ == "__main__":
    unittest.main()
