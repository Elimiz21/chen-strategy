"""
Data Versioning System
======================

Implements dataset versioning with SHA-256 hashes for reproducibility.
Every experiment must reference a specific data version.
"""

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


class DataVersioner:
    """
    Manage data versions for reproducibility.

    Stores metadata about each dataset version including:
    - Content hash (SHA-256)
    - Creation timestamp
    - Row/column counts
    - Date range
    """

    def __init__(self, registry_path: str = "data/versions.json"):
        """
        Initialize the versioner.

        Args:
            registry_path: Path to version registry file
        """
        self.registry_path = Path(registry_path)
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)

        self._registry = self._load_registry()

    def _load_registry(self) -> dict:
        """Load version registry from disk."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                return json.load(f)
        return {"versions": {}}

    def _save_registry(self):
        """Save version registry to disk."""
        with open(self.registry_path, "w") as f:
            json.dump(self._registry, f, indent=2, default=str)

    def compute_hash(self, df: pd.DataFrame) -> str:
        """
        Compute SHA-256 hash of DataFrame content.

        Args:
            df: DataFrame to hash

        Returns:
            16-character hash string
        """
        # Use CSV representation for consistent hashing
        data_bytes = df.to_csv().encode("utf-8")
        full_hash = hashlib.sha256(data_bytes).hexdigest()
        return full_hash[:16]

    def register_version(
        self,
        df: pd.DataFrame,
        name: str,
        description: str = "",
        source: str = "yahoo_finance",
    ) -> str:
        """
        Register a new data version.

        Args:
            df: DataFrame to register
            name: Human-readable name for this version
            description: Optional description
            source: Data source identifier

        Returns:
            Version hash
        """
        version_hash = self.compute_hash(df)

        # Check if already registered
        if version_hash in self._registry["versions"]:
            print(f"Version {version_hash} already registered")
            return version_hash

        # Extract metadata
        metadata = {
            "name": name,
            "description": description,
            "source": source,
            "created_at": datetime.now().isoformat(),
            "rows": len(df),
            "columns": list(df.columns),
            "date_range": {
                "start": str(df.index.min()),
                "end": str(df.index.max()),
            },
            "price_range": {
                "min": float(df["close"].min()) if "close" in df.columns else None,
                "max": float(df["close"].max()) if "close" in df.columns else None,
            },
        }

        self._registry["versions"][version_hash] = metadata
        self._save_registry()

        print(f"Registered data version: {version_hash}")
        print(f"  Name: {name}")
        print(f"  Rows: {metadata['rows']}")
        print(f"  Date range: {metadata['date_range']['start']} to {metadata['date_range']['end']}")

        return version_hash

    def get_version_info(self, version_hash: str) -> Optional[dict]:
        """
        Get metadata for a version.

        Args:
            version_hash: Version hash to look up

        Returns:
            Version metadata or None if not found
        """
        return self._registry["versions"].get(version_hash)

    def verify_version(self, df: pd.DataFrame, expected_hash: str) -> bool:
        """
        Verify that a DataFrame matches expected version.

        Args:
            df: DataFrame to verify
            expected_hash: Expected version hash

        Returns:
            True if hashes match
        """
        actual_hash = self.compute_hash(df)
        matches = actual_hash == expected_hash

        if matches:
            print(f"Version verified: {expected_hash}")
        else:
            print(f"Version mismatch! Expected {expected_hash}, got {actual_hash}")

        return matches

    def list_versions(self) -> list:
        """List all registered versions."""
        versions = []
        for hash_id, meta in self._registry["versions"].items():
            versions.append({
                "hash": hash_id,
                "name": meta["name"],
                "created_at": meta["created_at"],
                "rows": meta["rows"],
                "date_range": meta["date_range"],
            })
        return sorted(versions, key=lambda x: x["created_at"], reverse=True)

    def print_versions(self):
        """Print all registered versions."""
        versions = self.list_versions()

        if not versions:
            print("No versions registered")
            return

        print(f"\nRegistered Data Versions ({len(versions)} total):")
        print("-" * 80)

        for v in versions:
            print(f"  [{v['hash']}] {v['name']}")
            print(f"    Created: {v['created_at']}")
            print(f"    Rows: {v['rows']}")
            print(f"    Range: {v['date_range']['start']} to {v['date_range']['end']}")
            print()


if __name__ == "__main__":
    # Test the versioner
    from loader import load_qqq_data

    df = load_qqq_data()

    versioner = DataVersioner()
    version_hash = versioner.register_version(
        df,
        name="QQQ_full_history",
        description="Full QQQ history with technical indicators",
        source="yahoo_finance",
    )

    versioner.print_versions()

    # Verify
    versioner.verify_version(df, version_hash)
