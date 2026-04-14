"""
Sample dataset loader for MiniVecDB.

Loads and concatenates all dataset shards from the generated/ directory.
Each record includes text and metadata with category, subcategory, and source.
"""

import json
from pathlib import Path


def load_dataset():
    """
    Load all dataset shards and return a concatenated list of records.
    
    Returns:
        list: A list of dicts with keys 'text' and 'metadata'.
              Total: 150+ documents across 5 categories.
    """
    dataset_dir = Path(__file__).parent / "generated"
    
    # List of expected shard files (category order)
    shard_files = [
        "technology.json",
        "science.json",
        "sports.json",
        "health.json",
        "business.json",
    ]
    
    all_records = []
    
    for shard_file in shard_files:
        shard_path = dataset_dir / shard_file
        
        if shard_path.exists():
            with open(shard_path, "r") as f:
                shard_data = json.load(f)
                all_records.extend(shard_data)
        else:
            print(f"Warning: {shard_file} not found at {shard_path}")
    
    return all_records


if __name__ == "__main__":
    records = load_dataset()
    print(f"Loaded {len(records)} records")
    
    # Print category summary
    categories = {}
    for record in records:
        cat = record.get("metadata", {}).get("category", "Unknown")
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")
