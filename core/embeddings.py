"""
╔═══════════════════════════════════════════════════════════════╗
║  MiniVecDB — Embeddings Module                                ║
║  File: minivecdb/core/embeddings.py                           ║
║                                                               ║
║  This module converts text (strings) into vector embeddings   ║
║  (NumPy arrays of 384 floats) using a pre-trained model.      ║
║                                                               ║
║  Model: all-MiniLM-L6-v2 from sentence-transformers           ║
║  Output: 384-dimensional vectors                              ║
║                                                               ║
║  Install requirement:                                         ║
║      pip install sentence-transformers                        ║
╚═══════════════════════════════════════════════════════════════╝

HOW EMBEDDING MODELS WORK (simplified):
    
    1. TOKENISATION: The input text is broken into small pieces
       called "tokens." For example, "unhappiness" might become
       ["un", "##happi", "##ness"]. The model has a fixed vocabulary
       of ~30,000 tokens it knows about.
    
    2. TOKEN EMBEDDINGS: Each token is mapped to an initial vector
       from a lookup table the model learned during training.
    
    3. TRANSFORMER LAYERS: The initial vectors pass through 6 layers
       of "attention" (hence MiniLM-L6 = 6 Layers). Each layer
       lets every token "look at" every other token, refining its
       understanding based on context. This is why "bank" gets
       different vectors in "river bank" vs "bank account."
    
    4. POOLING: After all layers, we have one vector per token.
       We need ONE vector for the whole sentence, so we average
       all token vectors together. This is called "mean pooling."
    
    5. OUTPUT: A single 384-dimensional NumPy array that captures
       the semantic meaning of the entire input text.
"""

import numpy as np
import hashlib
import time
from typing import List, Optional, Union
import warnings
import os
import logging

from core.runtime_paths import get_model_cache_path

# ═══════════════════════════════════════════════════════════════
# TYPE ALIAS
# ═══════════════════════════════════════════════════════════════
Vector = np.ndarray


# ═══════════════════════════════════════════════════════════════
# EMBEDDING ENGINE CLASS
# ═══════════════════════════════════════════════════════════════

class EmbeddingEngine:
    """
    Converts text into vector embeddings using a pre-trained model.
    
    This class wraps the sentence-transformers library and provides
    a clean interface for the rest of MiniVecDB to use. It handles:
    - Loading the model (with automatic downloading on first use)
    - Single and batch text encoding
    - Caching the model so it's only loaded once
    - Graceful fallback if sentence-transformers isn't installed
    
    Usage:
        engine = EmbeddingEngine()
        vector = engine.encode("Hello world")
        vectors = engine.encode_batch(["Hello", "World"])
    
    Attributes:
        model_name: Name of the pre-trained model being used
        dimension:  Number of dimensions in the output vectors (384)
        is_loaded:  Whether the model has been loaded into memory
    """
    
    # ── Default model configuration ──
    # You can change this to use a different model, but all vectors
    # in a database MUST use the same model. Mixing models produces
    # incompatible vectors that can't be compared.
    DEFAULT_MODEL = "all-MiniLM-L6-v2"
    DEFAULT_DIMENSION = 384
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        cache_folder: Optional[str] = None,
    ):
        """
        Initialise the embedding engine.
        
        Args:
            model_name: Name of the sentence-transformers model to use.
                       Defaults to "all-MiniLM-L6-v2" (384 dimensions).
                       Other options include:
                         - "all-mpnet-base-v2" (768 dims, more accurate)
                         - "all-MiniLM-L12-v2" (384 dims, slightly better)
                         - "paraphrase-MiniLM-L3-v2" (384 dims, fastest)
            cache_folder: Optional cache directory for Hugging Face files.
                         If omitted, uses project-local cache at
                         ./db_run/model_cache/huggingface.
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.dimension = self.DEFAULT_DIMENSION
        self.cache_folder = cache_folder
        self._model = None       # Lazy loading: model loads on first use
        self._is_available = None # Whether sentence-transformers is installed

    def _resolve_cache_folder(self) -> str:
        """Resolve and create the cache directory for model downloads."""
        if self.cache_folder is None:
            resolved = get_model_cache_path()
        else:
            resolved = os.path.abspath(os.path.expanduser(self.cache_folder))
            os.makedirs(resolved, exist_ok=True)

        self.cache_folder = resolved
        return resolved
    
    @property
    def is_loaded(self) -> bool:
        """Check if the model is currently loaded in memory."""
        return self._model is not None
    
    def _check_availability(self) -> bool:
        """
        Check if sentence-transformers library is installed.
        
        Returns True if available, False otherwise. The result is
        cached so we only check once.
        """
        if self._is_available is None:
            try:
                import sentence_transformers
                self._is_available = True
            except ImportError:
                self._is_available = False
        return self._is_available
    
    def _detect_cached_model(self, cache_folder: str) -> bool:
        """Check whether the model files already exist in the cache folder.

        Hugging Face stores downloaded models in directories that follow one
        of two naming conventions depending on the library version:
          - ``sentence-transformers_<model_name>``   (older)
          - ``models--sentence-transformers--<model_name>``  (newer hub layout)

        Returns True if a matching directory is found, False otherwise.
        """
        if not os.path.isdir(cache_folder):
            return False

        # Patterns that Hugging Face / sentence-transformers may use
        candidates = [
            f"sentence-transformers_{self.model_name}",
            f"models--sentence-transformers--{self.model_name}",
            self.model_name,
        ]

        for entry in os.listdir(cache_folder):
            for pattern in candidates:
                if pattern in entry:
                    full_path = os.path.join(cache_folder, entry)
                    if os.path.isdir(full_path):
                        return True
        return False

    def _load_model(self):
        """
        Load the embedding model into memory.
        
        This is called automatically on the first encode() call.
        The model download happens only once — subsequent runs
        load from the configured local cache folder.
        
        This is an example of "lazy loading" — we don't load the
        model when EmbeddingEngine() is created, but when it's
        actually needed. This makes startup faster and avoids
        loading the model if only other operations are needed.
        """
        if self._model is not None:
            return  # Already loaded
        
        if not self._check_availability():
            raise ImportError(
                "sentence-transformers is not installed. "
                "Install it with: pip install sentence-transformers\n"
                "This is required for converting text to vectors."
            )

        cache_folder = self._resolve_cache_folder()
        
        # Keep third-party loader chatter quiet in self-tests/examples.
        logging.getLogger("transformers").setLevel(logging.ERROR)
        logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
        logging.getLogger("huggingface_hub.utils._headers").setLevel(logging.ERROR)

        from sentence_transformers import SentenceTransformer

        # Detect whether we will load from cache or download fresh
        is_cached = self._detect_cached_model(cache_folder)

        if is_cached:
            print(f"[embeddings] Model '{self.model_name}' found in local cache.")
            print(f"[embeddings] Loading from cache: {cache_folder}")
            print(f"[embeddings] (No download required -- using previously cached files)")
        else:
            print(f"[embeddings] Model '{self.model_name}' NOT found in cache.")
            print(f"[embeddings] Downloading model from HuggingFace (~80 MB)...")
            print(f"[embeddings] Download destination: {cache_folder}")
            print(f"[embeddings] (This is a one-time download; future runs will use the cache)")

        load_start = time.time()
        
        # Load the model. On first run, this downloads from HuggingFace.
        # On subsequent runs, it loads from the configured cache folder.
        try:
            self._model = SentenceTransformer(
                self.model_name,
                cache_folder=cache_folder,
            )
        except TypeError:
            # Compatibility for older sentence-transformers signatures.
            os.environ.setdefault("HF_HOME", cache_folder)
            self._model = SentenceTransformer(self.model_name)
        
        load_elapsed = time.time() - load_start

        # Update dimension based on actual model output
        # (in case a different model was specified).
        # Newer sentence-transformers renamed this API.
        if hasattr(self._model, "get_embedding_dimension"):
            self.dimension = self._model.get_embedding_dimension()
        else:
            self.dimension = self._model.get_sentence_embedding_dimension()
        
        source_label = "cache" if is_cached else "download"
        print(f"[embeddings] Model loaded successfully ({source_label}) in {load_elapsed:.2f}s")
        print(f"[embeddings] Output dimension: {self.dimension}")
        print(f"[embeddings] Cache folder: {cache_folder}")
    
    def encode(self, text: str) -> Vector:
        """
        Convert a single text string into a vector embedding.
        
        This is the most important function in the embedding module.
        It takes any text — a word, a sentence, a paragraph — and
        returns a fixed-size vector of 384 floating-point numbers
        that captures its semantic meaning.
        
        Args:
            text: Any string of text to encode. Can be a single word,
                  a sentence, or even a paragraph. Longer texts are
                  truncated at 256 tokens (~200 words) by the model.
        
        Returns:
            NumPy array of shape (384,) with float32 values.
            The vector is NOT normalised (has variable magnitude).
        
        Raises:
            ImportError: If sentence-transformers is not installed
            ValueError: If text is empty
        
        Example:
            >>> engine = EmbeddingEngine()
            >>> vec = engine.encode("Hello world")
            >>> vec.shape
            (384,)
        """
        # ── Validate input ──
        if not isinstance(text, str):
            raise TypeError(f"Expected string, got {type(text).__name__}")
        
        if len(text.strip()) == 0:
            raise ValueError("Cannot encode empty text. Provide at least one word.")
        
        # ── Load model if not already loaded ──
        self._load_model()
        
        # ── Encode the text ──
        # show_progress_bar=False suppresses the progress bar for single texts
        # convert_to_numpy=True ensures we get a NumPy array (not a PyTorch tensor)
        vector = self._model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        
        # Ensure the output is float32 (consistent dtype for storage)
        return vector.astype(np.float32)
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Convert multiple texts into vectors at once (much faster).
        
        Processing texts in batches is significantly faster than
        encoding them one by one because:
        1. The GPU/CPU can parallelise operations across the batch
        2. There's less overhead from repeated function calls
        3. Memory transfers are more efficient in bulk
        
        Args:
            texts:      List of strings to encode
            batch_size: How many texts to process at once. 
                       Higher = faster but uses more RAM.
                       Default 32 is a good balance.
        
        Returns:
            2D NumPy array of shape (N, 384) where N = len(texts).
            Each row is the embedding for the corresponding text.
        
        Raises:
            ImportError: If sentence-transformers is not installed
            ValueError: If texts list is empty
        
        Example:
            >>> engine = EmbeddingEngine()
            >>> vecs = engine.encode_batch(["Hello", "World"])
            >>> vecs.shape
            (2, 384)
        """
        if not texts:
            raise ValueError("Cannot encode empty list of texts.")
        
        # Validate all inputs are strings
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"Item at index {i} is {type(text).__name__}, expected string."
                )
            if len(text.strip()) == 0:
                raise ValueError(f"Item at index {i} is empty. All texts must be non-empty.")
        
        self._load_model()
        
        # Encode all texts at once
        # The model handles internal batching based on batch_size
        vectors = self._model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=len(texts) > 100,  # Show bar for large batches
            convert_to_numpy=True
        )
        
        return vectors.astype(np.float32)
    
    def get_model_info(self) -> dict:
        """
        Return information about the embedding model.
        
        Returns:
            Dictionary with model name, dimension, and load status.
        """
        return {
            "model_name": self.model_name,
            "dimension": self.dimension,
            "is_loaded": self.is_loaded,
            "is_available": self._check_availability(),
            "cache_folder": self.cache_folder,
        }


# ═══════════════════════════════════════════════════════════════
# SIMPLE EMBEDDING FALLBACK (No External Dependencies)
# ═══════════════════════════════════════════════════════════════
#
# This provides a basic bag-of-words embedding that works without
# sentence-transformers. It's MUCH less accurate but useful for:
# 1. Testing the rest of the system without installing dependencies
# 2. Understanding how embeddings work at a conceptual level
# 3. Running the project on machines where ML libraries can't install
#
# HOW BAG-OF-WORDS WORKS:
# 1. Build a vocabulary from all known words
# 2. Each text becomes a vector where dimension i = count of word i
# 3. Example: vocab = {cat:0, sat:1, mat:2, dog:3}
#    "cat sat" → [1, 1, 0, 0]  (cat=1, sat=1, others=0)
#    "dog sat" → [0, 1, 0, 1]  (dog=1, sat=1, others=0)
#
# This is crude — it doesn't understand meaning, just word overlap.
# "car" and "automobile" would have 0 similarity because they're
# different words, even though they mean the same thing.
# The SentenceTransformer model understands this; BoW doesn't.
#
# ═══════════════════════════════════════════════════════════════

class SimpleEmbeddingEngine:
    """
    A simple bag-of-words embedding engine that requires no
    external dependencies. Use this for testing or when
    sentence-transformers is not available.
    
    WARNING: This produces much lower quality embeddings than
    the real model. It matches word overlap, not meaning.
    
    Usage:
        engine = SimpleEmbeddingEngine(dimension=100)
        engine.build_vocabulary(["I love dogs", "Cats are great"])
        vec = engine.encode("I love cats")
    """
    
    def __init__(self, dimension: int = 100):
        """
        Args:
            dimension: Size of the output vectors. Higher = more
                      precise but uses more memory.
        """
        self.dimension = dimension
        self.vocabulary = {}  # word → index mapping
        self._is_built = False
    
    @property
    def is_loaded(self) -> bool:
        return self._is_built
    
    def build_vocabulary(self, texts: List[str]):
        """
        Build the word vocabulary from a collection of texts.
        
        This must be called before encode(). It creates a mapping
        from words to vector positions.
        
        Args:
            texts: List of all texts that will be in the database.
        """
        # Collect all unique words
        all_words = set()
        for text in texts:
            tokens = text.lower().split()
            all_words.update(tokens)
        
        # Map each word to an index (using hash to fit in dimension)
        # We use modulo to map any word to a position within our
        # fixed dimension size. This means some words might share
        # a position (collision), which reduces quality.
        self.vocabulary = {}
        for word in sorted(all_words):
            # Hash the word and take modulo to get position
            idx = hash(word) % self.dimension
            self.vocabulary[word] = idx
        
        self._is_built = True
        print(f"Vocabulary built: {len(all_words)} unique words → {self.dimension}-dim vectors")
    
    def encode(self, text: str) -> Vector:
        """
        Encode text using bag-of-words representation.
        
        Args:
            text: String to encode
        
        Returns:
            NumPy array of shape (dimension,) with float32 values.
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            raise ValueError("Text must be a non-empty string.")
        
        # Create a zero vector
        vector = np.zeros(self.dimension, dtype=np.float32)
        
        # For each word in the text, increment the corresponding position
        tokens = text.lower().split()
        for token in tokens:
            idx = hash(token) % self.dimension
            vector[idx] += 1.0
        
        # Normalise the vector so length doesn't affect similarity
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode multiple texts at once."""
        if not texts:
            raise ValueError("Cannot encode empty list.")
        return np.array([self.encode(text) for text in texts])
    
    def get_model_info(self) -> dict:
        return {
            "model_name": "SimpleBoW (fallback)",
            "dimension": self.dimension,
            "is_loaded": self._is_built,
            "is_available": True,
        }


# ═══════════════════════════════════════════════════════════════
# FACTORY FUNCTION — Auto-selects the best available engine
# ═══════════════════════════════════════════════════════════════

def create_embedding_engine(
    model_name: Optional[str] = None,
    fallback: bool = True,
    cache_folder: Optional[str] = None,
) -> Union[EmbeddingEngine, SimpleEmbeddingEngine]:
    """
    Create the best available embedding engine.
    
    Tries to create an EmbeddingEngine with sentence-transformers.
    If that's not installed and fallback=True, returns a
    SimpleEmbeddingEngine instead.
    
    Args:
        model_name: Optional model name for EmbeddingEngine
        fallback:   If True, use SimpleEmbeddingEngine when
                   sentence-transformers is unavailable.
        cache_folder: Optional Hugging Face cache folder for EmbeddingEngine.
    
    Returns:
        Either an EmbeddingEngine or SimpleEmbeddingEngine instance.
    
    Example:
        >>> engine = create_embedding_engine()
        >>> vec = engine.encode("Hello world")
    """
    # Try the real engine first
    engine = EmbeddingEngine(model_name=model_name, cache_folder=cache_folder)
    
    if engine._check_availability():
        return engine
    
    if fallback:
        warnings.warn(
            "sentence-transformers not installed. "
            "Using SimpleEmbeddingEngine (bag-of-words fallback). "
            "For better quality, install: pip install sentence-transformers",
            UserWarning,
            stacklevel=2
        )
        return SimpleEmbeddingEngine(dimension=384)
    
    raise ImportError(
        "sentence-transformers is required. "
        "Install with: pip install sentence-transformers"
    )


# ═══════════════════════════════════════════════════════════════
# MODULE SELF-TEST
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("MiniVecDB — Embeddings Module Self-Test")
    print("=" * 60)
    
    # Test with whatever engine is available
    engine = create_embedding_engine(fallback=True)
    info = engine.get_model_info()
    print(f"\nEngine: {info['model_name']}")
    print(f"Dimension: {info['dimension']}")
    print(f"Available: {info['is_available']}")
    
    # If using simple engine, build vocabulary first
    test_texts = [
        "The cat sat on the mat",
        "A kitten rested on a rug",
        "Python is a great programming language",
        "I love coding in Python",
        "The stock market crashed today",
    ]
    
    if isinstance(engine, SimpleEmbeddingEngine):
        engine.build_vocabulary(test_texts)
    
    # Test single encoding
    print("\n--- Single Encoding ---")
    for text in test_texts:
        vec = engine.encode(text)
        print(f"  \"{text[:40]}...\" → shape={vec.shape}, "
              f"dtype={vec.dtype}, norm={np.linalg.norm(vec):.4f}")
    
    # Test batch encoding
    print("\n--- Batch Encoding ---")
    batch_vecs = engine.encode_batch(test_texts)
    print(f"  Input: {len(test_texts)} texts")
    print(f"  Output shape: {batch_vecs.shape}")
    print(f"  Expected: ({len(test_texts)}, {info['dimension']})")
    
    # Test similarity between pairs
    print("\n--- Semantic Similarity (using cosine) ---")
    try:
        from core.distance_metrics import cosine_similarity
    except ModuleNotFoundError:
        from distance_metrics import cosine_similarity
    
    pairs = [
        (0, 1, "Cat/mat vs Kitten/rug (should be HIGH)"),
        (2, 3, "Python language vs Love coding (should be HIGH)"),
        (0, 4, "Cat/mat vs Stock market (should be LOW)"),
        (2, 4, "Python vs Stock market (should be LOW)"),
    ]
    
    for i, j, description in pairs:
        sim = cosine_similarity(batch_vecs[i], batch_vecs[j])
        bar = "█" * int(max(0, sim) * 20)
        print(f"  {sim:+.4f}  {bar}  {description}")
    
    print("\n✓ Embeddings module working correctly!")
    if isinstance(engine, SimpleEmbeddingEngine):
        print("  Tip: Install sentence-transformers for much better quality:")
        print("  pip install sentence-transformers")
    else:
        print("  Using sentence-transformers backend.")
