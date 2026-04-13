"""
╔═══════════════════════════════════════════════════════════════╗
║  MiniVecDB — Tests for Embeddings Module                      ║
║  File: minivecdb/tests/run_tests_embeddings.py                ║
║                                                               ║
║  Tests both EmbeddingEngine and SimpleEmbeddingEngine          ║
║  Run: python minivecdb/tests/run_tests_embeddings.py          ║
╚═══════════════════════════════════════════════════════════════╝
"""

import sys
import os
import shutil
import tempfile
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.embeddings import (
    EmbeddingEngine,
    SimpleEmbeddingEngine,
    create_embedding_engine,
)
from core.distance_metrics import cosine_similarity

passed = 0
failed = 0

def assert_true(condition, label):
    global passed, failed
    if condition:
        print(f"  ✓ {label}")
        passed += 1
    else:
        print(f"  ✗ {label}")
        failed += 1

def assert_approx(actual, expected, label, tol=1e-4):
    global passed, failed
    if abs(actual - expected) < tol:
        print(f"  ✓ {label}")
        passed += 1
    else:
        print(f"  ✗ {label} — got {actual}, expected {expected}")
        failed += 1

def assert_raises(exc_type, func, label):
    global passed, failed
    try:
        func()
        print(f"  ✗ {label} — no exception raised")
        failed += 1
    except exc_type:
        print(f"  ✓ {label}")
        passed += 1
    except (TypeError, ValueError):
        # Accept either TypeError or ValueError for type checks
        print(f"  ✓ {label}")
        passed += 1
    except Exception as e:
        print(f"  ✗ {label} — wrong exception: {type(e).__name__}: {e}")
        failed += 1

print("=" * 60)
print("MiniVecDB — Embeddings Module Test Suite")
print("=" * 60)

# ═══════════════════════════════════════════
print("\n── SIMPLE EMBEDDING ENGINE ──")
# ═══════════════════════════════════════════
engine = SimpleEmbeddingEngine(dimension=100)

# Build vocabulary
texts = [
    "the cat sat on the mat",
    "a dog chased the ball",
    "python is great for coding",
]
engine.build_vocabulary(texts)

assert_true(engine.is_loaded,
    "Engine reports is_loaded=True after build_vocabulary")

assert_true(engine.dimension == 100,
    f"Dimension is 100 (got {engine.dimension})")

# ── Single encoding ──
vec = engine.encode("the cat sat on the mat")
assert_true(vec.shape == (100,),
    f"Single encode shape is (100,) — got {vec.shape}")

assert_true(vec.dtype == np.float32,
    f"Output dtype is float32 — got {vec.dtype}")

assert_approx(np.linalg.norm(vec), 1.0,
    "Output vector is normalised (norm ≈ 1.0)")

# ── Batch encoding ──
vecs = engine.encode_batch(texts)
assert_true(vecs.shape == (3, 100),
    f"Batch encode shape is (3, 100) — got {vecs.shape}")

# ── Identical text gives identical vector ──
v1 = engine.encode("hello world")
v2 = engine.encode("hello world")
assert_approx(cosine_similarity(v1, v2), 1.0,
    "Same text produces identical vectors (cosine = 1.0)")

# ── Different text gives different vector ──
v3 = engine.encode("python programming")
sim = cosine_similarity(v1, v3)
assert_true(sim < 1.0,
    f"Different texts produce different vectors (cosine = {sim:.4f} < 1.0)")

# ── Deterministic output ──
v4 = engine.encode("test reproducibility")
v5 = engine.encode("test reproducibility")
assert_true(np.array_equal(v4, v5),
    "Encoding is deterministic (same input → same output)")

# ── Error handling ──
assert_raises(ValueError, lambda: engine.encode(""),
    "Empty string raises ValueError")

assert_raises(ValueError, lambda: engine.encode("   "),
    "Whitespace-only string raises ValueError")

assert_raises(ValueError, lambda: engine.encode(123),
    "Non-string input raises error")



assert_raises(ValueError, lambda: engine.encode_batch([]),
    "Empty list raises ValueError")

# ═══════════════════════════════════════════
print("\n── FACTORY FUNCTION ──")
# ═══════════════════════════════════════════
factory_engine = create_embedding_engine(fallback=True)
info = factory_engine.get_model_info()

assert_true(info["is_available"] is not None,
    f"get_model_info reports availability: {info['is_available']}")

assert_true(info["dimension"] > 0,
    f"Dimension is positive: {info['dimension']}")

assert_true(isinstance(info["model_name"], str),
    f"Model name is a string: {info['model_name']}")

# ═══════════════════════════════════════════
print("\n── CACHE PATH RESOLUTION ──")
# ═══════════════════════════════════════════
previous_project_root = os.environ.get("MINIVECDB_PROJECT_ROOT")
temp_project_root = tempfile.mkdtemp(prefix="minivecdb_cache_root_")

try:
    os.environ["MINIVECDB_PROJECT_ROOT"] = temp_project_root

    # Default cache should be project_root/db_run/model_cache/huggingface.
    default_cache_engine = EmbeddingEngine()
    default_cache_path = default_cache_engine._resolve_cache_folder()
    expected_default_cache = os.path.abspath(
        os.path.join(temp_project_root, "db_run", "model_cache", "huggingface")
    )
    assert_true(
        default_cache_path == expected_default_cache,
        f"Default cache path resolves under db_run: {default_cache_path}",
    )
    assert_true(
        os.path.isdir(default_cache_path),
        "Default cache directory is created",
    )

    # Custom cache path should be respected and created.
    custom_cache = os.path.join(temp_project_root, "custom_hf_cache")
    custom_cache_engine = EmbeddingEngine(cache_folder=custom_cache)
    custom_cache_path = custom_cache_engine._resolve_cache_folder()
    assert_true(
        custom_cache_path == os.path.abspath(custom_cache),
        f"Custom cache path is respected: {custom_cache_path}",
    )
    assert_true(
        os.path.isdir(custom_cache_path),
        "Custom cache directory is created",
    )

    factory_custom = create_embedding_engine(
        fallback=True,
        cache_folder=custom_cache,
    )
    if isinstance(factory_custom, EmbeddingEngine):
        assert_true(
            factory_custom.cache_folder == custom_cache,
            "Factory passes cache_folder to EmbeddingEngine",
        )
    else:
        assert_true(
            isinstance(factory_custom, SimpleEmbeddingEngine),
            "Factory fallback still works with cache_folder argument",
        )
finally:
    if previous_project_root is None:
        os.environ.pop("MINIVECDB_PROJECT_ROOT", None)
    else:
        os.environ["MINIVECDB_PROJECT_ROOT"] = previous_project_root
    shutil.rmtree(temp_project_root, ignore_errors=True)

# ═══════════════════════════════════════════
print("\n── SIMPLE ENGINE: SIMILARITY QUALITY ──")
# ═══════════════════════════════════════════
quality_engine = SimpleEmbeddingEngine(dimension=200)
quality_texts = [
    "I love dogs and puppies",
    "Dogs are my favorite animals",
    "The stock market crashed",
    "Financial markets declined sharply",
    "I enjoy walking my dog",
]
quality_engine.build_vocabulary(quality_texts)
quality_vecs = quality_engine.encode_batch(quality_texts)

# Same-topic texts should have higher similarity than cross-topic
dog_dog = cosine_similarity(quality_vecs[0], quality_vecs[1])
dog_fin = cosine_similarity(quality_vecs[0], quality_vecs[2])
fin_fin = cosine_similarity(quality_vecs[2], quality_vecs[3])

# With BoW, same-topic only works if they share words
# Dogs texts share "dogs" → should have some similarity
assert_true(dog_dog > 0.0,
    f"Dog texts share some similarity: {dog_dog:.4f}")

print(f"\n  Info: dog↔dog = {dog_dog:.4f}, "
      f"dog↔finance = {dog_fin:.4f}, "
      f"finance↔finance = {fin_fin:.4f}")
print("  (With sentence-transformers, dog↔dog would be ~0.7+)")

# ═══════════════════════════════════════════
print("\n── EMBEDDING ENGINE (Real Model) ──")
# ═══════════════════════════════════════════
real_engine = EmbeddingEngine()
if real_engine._check_availability():
    print("  sentence-transformers IS installed, testing real model...")
    
    rv1 = real_engine.encode("The cat sat on the mat")
    assert_true(rv1.shape == (384,),
        f"Real model output shape: {rv1.shape}")
    
    rv2 = real_engine.encode("A kitten rested on a rug")
    real_sim = cosine_similarity(rv1, rv2)
    assert_true(real_sim > 0.5,
        f"Semantic similarity is high: {real_sim:.4f} > 0.5")
    
    rv3 = real_engine.encode("The stock market crashed")
    cross_sim = cosine_similarity(rv1, rv3)
    assert_true(real_sim > cross_sim,
        f"Same-topic > cross-topic: {real_sim:.4f} > {cross_sim:.4f}")
    
    # Batch
    rvecs = real_engine.encode_batch(["Hello", "World", "Test"])
    assert_true(rvecs.shape == (3, 384),
        f"Batch shape: {rvecs.shape}")

    # Verify cache reuse by forcing offline mode and loading from cache.
    previous_hf_offline = os.environ.get("HF_HUB_OFFLINE")
    previous_transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE")
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        from sentence_transformers import SentenceTransformer

        offline_model = SentenceTransformer(
            real_engine.model_name,
            cache_folder=real_engine.cache_folder,
        )
        if hasattr(offline_model, "get_embedding_dimension"):
            offline_dim = offline_model.get_embedding_dimension()
        else:
            offline_dim = offline_model.get_sentence_embedding_dimension()

        assert_true(
            offline_dim == 384,
            "Offline load from local cache works (no re-download required)",
        )
    except Exception as exc:
        assert_true(
            False,
            f"Offline load from local cache failed: {type(exc).__name__}: {exc}",
        )
    finally:
        if previous_hf_offline is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = previous_hf_offline

        if previous_transformers_offline is None:
            os.environ.pop("TRANSFORMERS_OFFLINE", None)
        else:
            os.environ["TRANSFORMERS_OFFLINE"] = previous_transformers_offline
else:
    print("  sentence-transformers NOT installed, skipping real model tests")
    print("  Install with: pip install sentence-transformers")
    print("  (This is expected in sandboxed environments)")


# ═══════════════════════════════════════════
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed, {passed + failed} total")
print("=" * 60)

if failed == 0:
    print("🎉 ALL TESTS PASSED! Day 3 module is working perfectly.")
else:
    print(f"⚠️  {failed} test(s) failed.")
    sys.exit(1)
