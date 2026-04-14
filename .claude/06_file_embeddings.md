# MiniVecDB — File: core/embeddings.py

> **Location**: `minivecdb/core/embeddings.py`
> **Lines**: 568 | **Size**: 23.4 KB
> **Purpose**: Converts text strings into 384-dimensional vector embeddings using a neural model

---

## Why This File Exists

A vector database is useless without a way to convert text into vectors. This module provides two embedding engines:

1. **`EmbeddingEngine`** — Uses the `all-MiniLM-L6-v2` neural model for high-quality semantic embeddings
2. **`SimpleEmbeddingEngine`** — A bag-of-words fallback that works without any ML dependencies

Plus a **factory function** that auto-selects the best available engine.

---

## How Embedding Models Work (Explained in the File)

The file header (lines 16-38) contains an excellent teaching explanation:

```
1. TOKENISATION    → "unhappiness" → ["un", "##happi", "##ness"]
2. TOKEN EMBEDDINGS → Each token → initial vector from lookup table
3. TRANSFORMER LAYERS → 6 layers of "attention" refine understanding
4. POOLING         → Average all token vectors → one sentence vector
5. OUTPUT          → 384-dimensional float32 array
```

The key insight: "bank" gets different vectors in "river bank" vs "bank account" because the attention mechanism considers context.

---

## Classes & Functions

### Class: `EmbeddingEngine`
**Lines 59–321** | The production-quality embedding engine.

#### Attributes
| Attribute | Type | Purpose |
|-----------|------|---------|
| `model_name` | str | `"all-MiniLM-L6-v2"` (default) |
| `dimension` | int | `384` (output vector size) |
| `cache_folder` | str | Where model weights are cached on disk |
| `_model` | SentenceTransformer | The loaded model (None until first use) |
| `_is_available` | bool | Whether sentence-transformers is installed |

#### `__init__(model_name=None, cache_folder=None)`
**Lines 88–111**

Initialises configuration but does **NOT** load the model. This is **lazy loading** — the model only loads when you first call `encode()`. Benefits:
- Faster startup (no 80MB model load if you're just doing metadata operations)
- The import won't fail if sentence-transformers is installed but the model isn't cached yet

#### `_resolve_cache_folder() → str`
**Lines 113–122**

Determines where to store the downloaded model:
- If `cache_folder` was explicitly provided, use that
- Otherwise, call `get_model_cache_path()` to use `db_run/model_cache/huggingface/`

This keeps the 80MB model cache *inside the project* rather than polluting `~/.cache/`.

#### `_check_availability() → bool`
**Lines 129–142**

Tries `import sentence_transformers` in a try/except. The result is cached so we only check once.

#### `_load_model()`
**Lines 144–201** | The lazy model loader.

What happens step-by-step:
1. Return immediately if already loaded
2. Check if sentence-transformers is installed (raise ImportError if not)
3. Resolve the cache folder
4. Suppress noisy log output from transformers/huggingface_hub
5. Import `SentenceTransformer` and create the model instance
6. On first run: downloads ~80MB from HuggingFace (cached for future runs)
7. Update `self.dimension` from the actual model output (in case a non-default model was specified)

**Compatibility note**: Newer versions of sentence-transformers renamed `get_sentence_embedding_dimension()` to `get_embedding_dimension()`. The code tries both.

#### `encode(text: str) → Vector`
**Lines 203–251** | The most important function — converts ONE text to ONE vector.

**Algorithm**:
1. Validate input is a non-empty string
2. Call `_load_model()` (lazy load on first use)
3. Call `self._model.encode(text, show_progress_bar=False, convert_to_numpy=True)`
4. Cast to `float32` (ensures consistent dtype for storage)

**Returns**: NumPy array of shape `(384,)` with `float32` dtype.

#### `encode_batch(texts: List[str], batch_size=32) → np.ndarray`
**Lines 253–306** | Converts MULTIPLE texts at once — **much faster**.

**Why batch is faster**:
1. GPU/CPU parallelises across the batch
2. Less Python overhead from repeated function calls
3. Memory transfers are more efficient in bulk

**Algorithm**:
1. Validate all inputs are non-empty strings
2. Lazy-load model
3. Call `self._model.encode(texts, batch_size=32, show_progress_bar=..., convert_to_numpy=True)`
4. Cast to `float32`

**Returns**: NumPy array of shape `(N, 384)` where N = number of texts.

#### `get_model_info() → dict`
**Lines 308–321** | Returns metadata about the engine (name, dimension, load status, availability, cache path).

---

### Class: `SimpleEmbeddingEngine`
**Lines 348–447** | The fallback engine that requires no external dependencies.

This is a **bag-of-words** embedding that:
1. Builds a vocabulary of known words
2. Maps each word to a vector position using a hash function
3. Counts word occurrences to create a sparse vector
4. Normalises the vector to unit length

**Quality**: Much lower than the neural model. "car" and "automobile" would have 0 similarity despite being synonyms, because they're different words. The neural model understands this; BoW doesn't.

#### Key Functions

| Function | What It Does |
|----------|-------------|
| `build_vocabulary(texts)` | Extracts unique words and maps them to positions via `hash(word) % dimension` |
| `encode(text)` | Creates a zero vector, increments positions for each word, normalises |
| `encode_batch(texts)` | Calls `encode()` in a loop (no GPU parallelism) |

**Hash collision note**: Two different words might hash to the same position (collision), reducing quality. This is acceptable for a fallback.

---

### Factory Function: `create_embedding_engine()`
**Lines 454–498**

```python
def create_embedding_engine(model_name=None, fallback=True, cache_folder=None):
```

**Decision logic**:
1. Create an `EmbeddingEngine`
2. Check if sentence-transformers is available
3. If yes → return the real engine
4. If no and `fallback=True` → issue a warning, return `SimpleEmbeddingEngine(dimension=384)`
5. If no and `fallback=False` → raise `ImportError`

This factory is called in `VectorStore.__init__()` with `fallback=True`, ensuring the system always works even without ML dependencies.

---

## Self-Test (Lines 505–567)

When run directly, tests:
1. Creating the best available engine
2. Single text encoding (5 test texts)
3. Batch encoding (all texts at once)
4. Semantic similarity between text pairs using cosine similarity
5. Compares "cat/mat" vs "kitten/rug" (should be HIGH similarity) and "cat/mat" vs "stock market" (should be LOW similarity)
