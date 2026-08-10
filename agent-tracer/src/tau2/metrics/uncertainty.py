"""
Uncertainty Metrics for TRACER Framework

This module implements uncertainty quantification metrics for the TRACER
(Situation-Awareness Uncertainty Propagation) framework on Tau-2 benchmark.

Key metrics:
- Normalized Entropy (single-step uncertainty)
- Token-level uncertainties
- Response statistics
- Semantic Distance Metrics:
  - Inquiry Drift (Da): measures deviation from initial goal
  - Inference Gap (Do): measures coherence between action and observation
- TRACER Aggregation: weighted RMS combining all metrics into trajectory score
"""

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger
from pydantic import BaseModel

# Vertex AI imports for embeddings
try:
    import vertexai
    from vertexai.language_models import TextEmbeddingModel
    VERTEX_AI_AVAILABLE = True
except ImportError:
    VERTEX_AI_AVAILABLE = False
    logger.warning("Vertex AI not available. Semantic distance metrics will be disabled.")


class TokenUncertainty(BaseModel):
    """Token-level uncertainty information."""

    token: str
    logprob: float
    neg_log_likelihood: float
    probability: float


class UncertaintyStats(BaseModel):
    """Comprehensive uncertainty statistics for a response."""

    normalized_entropy: float
    total_entropy: float
    token_count: int
    min_uncertainty: float
    max_uncertainty: float
    std_uncertainty: float
    mean_probability: float


# Stop words and structural tokens to filter from entropy calculation
STOP_WORDS = {
    # Common stop words
    "the", "is", "are", "was", "were", "be", "been", "being",
    "a", "an", "to", "of", "in", "on", "at", "for", "with",
    "from", "by", "as", "or", "and", "but", "if", "then",
    "this", "that", "these", "those", "it", "its", "i", "you",
    "he", "she", "we", "they", "my", "your", "his", "her",
    "am", "can", "will", "would", "could", "should", "may",
    "have", "has", "had", "do", "does", "did",
    # Filler phrases (split into tokens)
    "please", "sorry", "wait", "help", "need", "want",
    # Structural/formatting tokens
    "[tool_call]", "\\n", "\n", " ", "", ".", ",", "!", "?",
    ":", ";", "-", "(", ")", "[", "]", "{", "}", "'", '"',
}


def calculate_normalized_entropy(logprobs_object: Optional[dict]) -> float:
    """
    Calculate the Normalized Entropy (single-step uncertainty) for a response.

    This metric is defined as the average token-level negative log-likelihood
    of the generated response, focusing on CONTENT-BEARING tokens only.
    
    Filtering Strategy (fixes entropy dilution):
    1. Filters high-probability tokens (P > 95%) - likely structural
    2. Filters stop words and filler phrases - masks true uncertainty
    3. Measures uncertainty only on entities, actions, and domain-specific terms

    Mathematically:
        U_i = (1/|R_i|) * Σ(-log P(token_j)) for content tokens only

    Where:
        - |R_i| is the number of CONTENT tokens in the response
        - -log P(token_j) is the negative log-likelihood of token j

    Args:
        logprobs_object: Dictionary containing logprobs data from the LLM API.
                        Expected structure:
                        {
                            'content': [
                                {'token': str, 'logprob': float, ...},
                                ...
                            ]
                        }
                        Can be None if logprobs are not available.

    Returns:
        float: The normalized entropy (average negative log-likelihood).
               Returns 1e-4 if:
               - logprobs_object is None
               - No tokens are found
               - The content array is empty
               - All tokens were filtered (response is only filler)

    Example:
        >>> logprobs = {
        ...     'content': [
        ...         {'token': 'I', 'logprob': -0.01},  # Stop word, filtered
        ...         {'token': ' am', 'logprob': -0.01},  # Stop word, filtered
        ...         {'token': ' booking', 'logprob': -1.2},  # Content token
        ...         {'token': ' flight', 'logprob': -0.8},  # Content token
        ...     ]
        ... }
        >>> calculate_normalized_entropy(logprobs)
        1.0  # (1.2 + 0.8) / 2 - only content tokens measured
    """
    # Handle None or missing logprobs
    if logprobs_object is None:
        return 1e-4

    # Extract the content array
    content = logprobs_object.get("content", [])

    if not content or len(content) == 0:
        return 1e-4

    # Initialize counters
    total_neg_log_likelihood = 0.0
    token_count = 0

    # Iterate through each token in the sequence
    for token_data in content:
        # Extract the logprob value
        logprob = token_data.get("logprob")
        token = token_data.get("token", "")

        if logprob is not None:
            # Filter 1: High-probability structural tokens (P > 95%)
            # logprob > -0.05 means P(token) > exp(-0.05) ≈ 0.951
            if logprob > -0.05:
                continue  # Skip structural tokens like newlines, spaces, etc.
            
            # Filter 2: Stop words and filler phrases
            # Clean token: strip whitespace, lowercase, remove punctuation
            cleaned_token = token.strip().lower()
            # Remove leading/trailing punctuation
            cleaned_token = cleaned_token.strip('.,!?;:()[]{}"\'-')
            
            # Skip if empty after cleaning
            if not cleaned_token:
                continue
            
            # Skip if it's a stop word
            if cleaned_token in STOP_WORDS:
                continue
            
            # Skip if it's purely numeric (dates, numbers, etc.)
            if cleaned_token.replace('.', '').replace('-', '').isdigit():
                continue
            
            # This is a content-bearing token - accumulate its uncertainty
            neg_log_likelihood = -logprob
            total_neg_log_likelihood += neg_log_likelihood
            token_count += 1

    # Avoid division by zero - if all tokens were filtered, return small epsilon
    # This indicates the response is 100% filler (a failure signal itself)
    if token_count == 0:
        return 1e-4

    # Calculate and return the normalized entropy (average)
    normalized_entropy = total_neg_log_likelihood / token_count

    return normalized_entropy


def calculate_entropy(logprobs_object: Optional[dict]) -> float:
    """
    Calculate the standard entropy (unnormalized) for a response.

    This is the total negative log-likelihood across all tokens in the response,
    without normalization by token count.

    Args:
        logprobs_object: Dictionary containing logprobs data from the LLM API.

    Returns:
        float: The total entropy (sum of negative log-likelihoods).
    """
    if logprobs_object is None:
        return 0.0

    content = logprobs_object.get("content", [])

    if not content or len(content) == 0:
        return 0.0

    total_neg_log_likelihood = 0.0

    for token_data in content:
        logprob = token_data.get("logprob")
        if logprob is not None:
            total_neg_log_likelihood += -logprob

    return total_neg_log_likelihood


def calculate_token_uncertainties(
    logprobs_object: Optional[dict],
) -> list[TokenUncertainty]:
    """
    Calculate token-level uncertainties for detailed analysis.

    Returns uncertainty information for each individual token in the response.

    Args:
        logprobs_object: Dictionary containing logprobs data from the LLM API.

    Returns:
        list[TokenUncertainty]: List of token uncertainty objects.
    """
    if logprobs_object is None:
        return []

    content = logprobs_object.get("content", [])

    if not content:
        return []

    token_uncertainties = []

    for token_data in content:
        token = token_data.get("token", "")
        logprob = token_data.get("logprob")

        if logprob is not None:
            token_uncertainties.append(
                TokenUncertainty(
                    token=token,
                    logprob=logprob,
                    neg_log_likelihood=-logprob,
                    probability=np.exp(logprob),
                )
            )

    return token_uncertainties


def get_uncertainty_stats(logprobs_object: Optional[dict]) -> UncertaintyStats:
    """
    Get comprehensive statistics about the response uncertainty.

    Args:
        logprobs_object: Dictionary containing logprobs data from the LLM API.

    Returns:
        UncertaintyStats: Comprehensive uncertainty statistics.
    """
    if logprobs_object is None:
        return UncertaintyStats(
            normalized_entropy=0.0,
            total_entropy=0.0,
            token_count=0,
            min_uncertainty=0.0,
            max_uncertainty=0.0,
            std_uncertainty=0.0,
            mean_probability=0.0,
        )

    content = logprobs_object.get("content", [])

    if not content:
        return UncertaintyStats(
            normalized_entropy=0.0,
            total_entropy=0.0,
            token_count=0,
            min_uncertainty=0.0,
            max_uncertainty=0.0,
            std_uncertainty=0.0,
            mean_probability=0.0,
        )

    uncertainties = []
    probabilities = []

    for token_data in content:
        logprob = token_data.get("logprob")
        if logprob is not None:
            uncertainties.append(-logprob)
            probabilities.append(np.exp(logprob))

    if not uncertainties:
        return UncertaintyStats(
            normalized_entropy=0.0,
            total_entropy=0.0,
            token_count=0,
            min_uncertainty=0.0,
            max_uncertainty=0.0,
            std_uncertainty=0.0,
            mean_probability=0.0,
        )

    return UncertaintyStats(
        normalized_entropy=float(np.mean(uncertainties)),
        total_entropy=float(np.sum(uncertainties)),
        token_count=len(uncertainties),
        min_uncertainty=float(np.min(uncertainties)),
        max_uncertainty=float(np.max(uncertainties)),
        std_uncertainty=float(np.std(uncertainties)),
        mean_probability=float(np.mean(probabilities)),
    )


# ============================================================================
# Semantic Distance Metrics (TRACER Situational Awareness Layer)
# ============================================================================


class EmbeddingService:
    """
    Singleton service for managing Vertex AI text embeddings.
    
    Loads the embedding model once and reuses it for all semantic distance
    calculations during simulation.
    """
    
    _instance: Optional['EmbeddingService'] = None
    _model: Optional['TextEmbeddingModel'] = None
    _initialized: bool = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the embedding service (only once)."""
        if not self._initialized and VERTEX_AI_AVAILABLE:
            try:
                # Initialize Vertex AI (auto-detects project from environment)
                # User should have GOOGLE_APPLICATION_CREDENTIALS or gcloud auth set up
                project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
                if project_id:
                    vertexai.init(project=project_id)
                else:
                    # Try default initialization
                    vertexai.init()
                
                # Load the embedding model
                # Using text-embedding-004 (latest, 768 dimensions)
                self._model = TextEmbeddingModel.from_pretrained("text-embedding-004")
                self._initialized = True
                logger.info("✓ Vertex AI embedding service initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize Vertex AI embeddings: {e}")
                self._model = None
                self._initialized = False
    
    def get_embedding(self, text: str) -> Optional[np.ndarray]:
        """
        Generate embedding for a text string.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding vector as numpy array, or None if service unavailable
        """
        if not self._initialized or self._model is None:
            return None
        
        if not text or not text.strip():
            return None
        
        try:
            # Truncate text if too long (model limit is ~20k tokens)
            # Keep approximately 15k tokens worth of text
            max_chars = 60000
            if len(text) > max_chars:
                text = text[-max_chars:]  # Keep most recent content
                logger.debug(f"Truncated text from {len(text)} to {max_chars} chars")
            
            # Get embeddings
            embeddings = self._model.get_embeddings([text])
            if embeddings and len(embeddings) > 0:
                # Convert to numpy array
                return np.array(embeddings[0].values)
            return None
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None
    
    def is_available(self) -> bool:
        """Check if the embedding service is available."""
        return self._initialized and self._model is not None


def calculate_semantic_distance(text_a: str, text_b: str) -> float:
    """
    Calculate semantic distance between two texts using cosine similarity.
    
    The semantic distance is defined as:
        distance = 1 - cosine_similarity
    
    Where cosine_similarity is the cosine of the angle between two embedding
    vectors. A distance of 0 means the texts are semantically identical,
    while higher values indicate greater semantic divergence.
    
    Args:
        text_a: First text string
        text_b: Second text string
        
    Returns:
        float: Semantic distance in range [0, 2] where:
               - 0.0 = identical meaning
               - ~0.5 = moderately related
               - ~1.0 = unrelated
               - >1.0 = opposite meaning (rare)
               Returns 0.0 if embeddings cannot be generated
    
    Example:
        >>> distance = calculate_semantic_distance(
        ...     "Book a flight to Paris",
        ...     "Reserve tickets to France"
        ... )
        >>> # Should return a low value (similar meaning)
    """
    # Handle edge cases
    if not text_a or not text_b:
        return 0.0
    
    if not text_a.strip() or not text_b.strip():
        return 0.0
    
    # Get embeddings
    service = EmbeddingService()
    if not service.is_available():
        logger.debug("Embedding service not available, returning 0.0")
        return 0.0
    
    embedding_a = service.get_embedding(text_a)
    embedding_b = service.get_embedding(text_b)
    
    if embedding_a is None or embedding_b is None:
        logger.debug("Failed to generate embeddings, returning 0.0")
        return 0.0
    
    # Calculate cosine similarity
    try:
        # Normalize vectors
        norm_a = np.linalg.norm(embedding_a)
        norm_b = np.linalg.norm(embedding_b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        # Cosine similarity = dot product / (norm_a * norm_b)
        cosine_sim = np.dot(embedding_a, embedding_b) / (norm_a * norm_b)
        
        # Distance = 1 - similarity
        distance = 1.0 - float(cosine_sim)
        
        # Clamp to [0, 2] range (theoretical maximum)
        distance = max(0.0, min(2.0, distance))
        
        return distance
    except Exception as e:
        logger.error(f"Error calculating cosine similarity: {e}")
        return 0.0


def calculate_jaccard_similarity(text_a: str, text_b: str) -> float:
    """
    Calculate Jaccard similarity between two texts (lexical overlap).
    
    Measures the intersection-over-union of content tokens, filtering out
    stop words and punctuation. This captures whether the specific entities
    and actions are identical, not just the semantic intent.
    
    Formula:
        Jaccard = |tokens_a ∩ tokens_b| / |tokens_a ∪ tokens_b|
    
    Args:
        text_a: First text string
        text_b: Second text string
        
    Returns:
        float: Jaccard similarity in range [0, 1] where:
               - 0.0 = no token overlap
               - 0.5 = moderate overlap
               - 1.0 = identical token sets
               Returns 0.0 if either text is empty or all tokens filtered
    
    Example:
        >>> # Looping case
        >>> jaccard = calculate_jaccard_similarity(
        ...     "I can help with that",
        ...     "I can help with that"
        ... )
        >>> # Returns ~1.0 (identical tokens)
        
        >>> # Enumeration case
        >>> jaccard = calculate_jaccard_similarity(
        ...     "Flight A123 is cancelled",
        ...     "Flight B456 is cancelled"
        ... )
        >>> # Returns ~0.33 (only "Flight" and "cancelled" overlap)
    """
    # Handle edge cases
    if not text_a or not text_b:
        return 0.0
    
    if not text_a.strip() or not text_b.strip():
        return 0.0
    
    # Tokenize: split by whitespace
    tokens_a_raw = text_a.lower().split()
    tokens_b_raw = text_b.lower().split()
    
    # Clean tokens: remove stop words and punctuation
    def clean_token(token: str) -> str:
        """Remove leading/trailing punctuation from token."""
        return token.strip('.,!?;:()[]{}"\'-')
    
    # Filter tokens
    tokens_a = set()
    for token in tokens_a_raw:
        cleaned = clean_token(token)
        # Skip if empty, stop word, or purely numeric
        if not cleaned or cleaned in STOP_WORDS:
            continue
        if cleaned.replace('.', '').replace('-', '').isdigit():
            continue
        tokens_a.add(cleaned)
    
    tokens_b = set()
    for token in tokens_b_raw:
        cleaned = clean_token(token)
        if not cleaned or cleaned in STOP_WORDS:
            continue
        if cleaned.replace('.', '').replace('-', '').isdigit():
            continue
        tokens_b.add(cleaned)
    
    # Handle case where all tokens were filtered
    if not tokens_a or not tokens_b:
        return 0.0
    
    # Calculate Jaccard similarity
    intersection = len(tokens_a & tokens_b)
    union = len(tokens_a | tokens_b)
    
    if union == 0:
        return 0.0
    
    jaccard_sim = float(intersection) / float(union)
    
    return jaccard_sim


def calculate_hybrid_repetition_score(current_text: str, history_texts: list[str]) -> float:
    """
    Calculate Hybrid Repetition score - combines semantic and lexical similarity.
    
    This metric solves the "Enumeration vs. Looping" problem:
    - **Looping:** High semantic + high lexical → High penalty (stuck repeating)
    - **Enumeration:** High semantic + low lexical → Low penalty (valid iteration)
    
    Formula:
        Hybrid = max(CosineSim_i × JaccardSim_i) for last 3 turns
    
    The multiplication acts as a "soft AND" gate: both semantic intent AND
    specific tokens must match for a high repetition score.
    
    Args:
        current_text: Current agent response text
        history_texts: List of previous agent response texts
                       
    Returns:
        float: Hybrid repetition score in range [0, 1] where:
               - 0.0 = novel response (different intent OR different entities)
               - ~0.5 = partial repetition
               - >0.8 = true looping (same intent AND same entities)
               Returns 0.0 if history is empty or embeddings unavailable
    
    Example (Looping - High Penalty):
        >>> history = ["I can help with that", "Let me assist you"]
        >>> current = "I can help with that"  # Exact repeat
        >>> score = calculate_hybrid_repetition_score(current, history)
        >>> # Returns ~1.0 (high cosine × high jaccard)
    
    Example (Enumeration - Low Penalty):
        >>> history = ["Flight A123 cancelled", "Flight B456 cancelled"]
        >>> current = "Flight C789 cancelled"  # Different entity
        >>> score = calculate_hybrid_repetition_score(current, history)
        >>> # Returns ~0.3 (high cosine × low jaccard)
    """
    # Handle edge cases
    if not current_text or not current_text.strip():
        return 0.0
    
    if not history_texts:
        return 0.0
    
    # Get embedding service for semantic similarity
    service = EmbeddingService()
    if not service.is_available():
        logger.debug("Embedding service not available, falling back to Jaccard only")
        # Fallback: use Jaccard similarity only
        max_jaccard = 0.0
        recent_history = history_texts[-3:] if len(history_texts) > 3 else history_texts
        for past_text in recent_history:
            jaccard = calculate_jaccard_similarity(current_text, past_text)
            max_jaccard = max(max_jaccard, jaccard)
        return max_jaccard
    
    # Get embedding for current text
    current_embedding = service.get_embedding(current_text)
    if current_embedding is None:
        return 0.0
    
    # Look at last 3 turns (local window for loop detection)
    recent_history = history_texts[-3:] if len(history_texts) > 3 else history_texts
    
    max_hybrid_score = 0.0
    
    for past_text in recent_history:
        if not past_text or not past_text.strip():
            continue
        
        # Calculate semantic similarity (cosine)
        past_embedding = service.get_embedding(past_text)
        if past_embedding is None:
            continue
        
        try:
            norm_current = np.linalg.norm(current_embedding)
            norm_past = np.linalg.norm(past_embedding)
            
            if norm_current == 0 or norm_past == 0:
                continue
            
            # Cosine similarity
            cosine_sim = np.dot(current_embedding, past_embedding) / (norm_current * norm_past)
            cosine_sim = max(0.0, min(1.0, float(cosine_sim)))
            
            # Calculate lexical similarity (Jaccard)
            jaccard_sim = calculate_jaccard_similarity(current_text, past_text)
            
            # Hybrid score: soft AND gate
            # Both intent AND tokens must match for high score
            hybrid_score = cosine_sim * jaccard_sim
            
            # Track maximum (worst-case repetition)
            max_hybrid_score = max(max_hybrid_score, hybrid_score)
            
            logger.debug(
                f"Repetition check: cosine={cosine_sim:.3f}, jaccard={jaccard_sim:.3f}, "
                f"hybrid={hybrid_score:.3f}"
            )
        except Exception as e:
            logger.error(f"Error calculating hybrid repetition: {e}")
            continue
    
    return max_hybrid_score


def calculate_tool_repetition(
    current_tool_calls: list,
    tool_call_history: list[list]
) -> float:
    """
    Calculate Tool Repetition penalty for exact duplicate tool calls.
    
    Exact duplicate tool calls (same name + same arguments) are always logical
    failures - the agent is stuck retrying the same failed operation.
    
    Formula:
        ToolRepetition = 1.0 if exact duplicate found in recent history, else 0.0
    
    Args:
        current_tool_calls: List of current tool call objects/dicts with
                           'name' and 'arguments' fields
        tool_call_history: List of past tool call lists (from last 5 turns)
                          Each element is a list of tool calls from one turn
                       
    Returns:
        float: Tool repetition score:
               - 1.0 = exact duplicate found (failure signal)
               - 0.0 = no duplicates (valid)
    
    Example (Duplicate - High Penalty):
        >>> current = [{'name': 'get_flight', 'arguments': {'id': 'A123'}}]
        >>> history = [
        ...     [{'name': 'get_flight', 'arguments': {'id': 'A123'}}],
        ...     [{'name': 'cancel_flight', 'arguments': {'id': 'B456'}}]
        ... ]
        >>> score = calculate_tool_repetition(current, history)
        >>> # Returns 1.0 (exact duplicate in history)
    
    Example (Different Args - No Penalty):
        >>> current = [{'name': 'get_flight', 'arguments': {'id': 'B456'}}]
        >>> history = [
        ...     [{'name': 'get_flight', 'arguments': {'id': 'A123'}}]
        ... ]
        >>> score = calculate_tool_repetition(current, history)
        >>> # Returns 0.0 (different arguments)
    """
    if not current_tool_calls:
        return 0.0
    
    if not tool_call_history:
        return 0.0
    
    # Create signatures for current tool calls
    def create_tool_signature(tool_call) -> str:
        """Create a unique signature for a tool call."""
        try:
            # Handle both dict and object formats
            if isinstance(tool_call, dict):
                name = tool_call.get('name', '')
                arguments = tool_call.get('arguments', {})
            else:
                name = getattr(tool_call, 'name', '')
                arguments = getattr(tool_call, 'arguments', {})
            
            # Sort arguments for consistent hashing
            if isinstance(arguments, dict):
                sorted_args = sorted(arguments.items())
            elif isinstance(arguments, str):
                # Arguments might be JSON string
                try:
                    import json
                    args_dict = json.loads(arguments)
                    sorted_args = sorted(args_dict.items())
                except:
                    sorted_args = [(arguments,)]
            else:
                sorted_args = [(str(arguments),)]
            
            # Create signature: name + sorted arguments
            signature = f"{name}:{sorted_args}"
            return signature
        except Exception as e:
            logger.debug(f"Error creating tool signature: {e}")
            return ""
    
    # Get signatures for current tool calls
    current_signatures = set()
    for tool_call in current_tool_calls:
        sig = create_tool_signature(tool_call)
        if sig:
            current_signatures.add(sig)
    
    if not current_signatures:
        return 0.0
    
    # Check against recent history (last 5 turns)
    recent_history = tool_call_history[-5:] if len(tool_call_history) > 5 else tool_call_history
    
    for past_tool_calls in recent_history:
        if not past_tool_calls:
            continue
        
        # Get signatures for this historical turn
        for past_tool_call in past_tool_calls:
            past_sig = create_tool_signature(past_tool_call)
            if past_sig and past_sig in current_signatures:
                # Found exact duplicate!
                logger.debug(f"Tool repetition detected: {past_sig}")
                return 1.0
    
    return 0.0


def calculate_inquiry_drift(
    user_instruction: str,
    history_so_far: list[str]
) -> float:
    """
    Calculate Inquiry Drift (Da) - semantic distance from initial goal.
    
    **DEPRECATED:** This metric is flawed for task-oriented dialogue
    
    Measures how much the conversation has drifted from the original user
    instruction/goal. This is computed by comparing the initial instruction
    with the concatenated conversation history.
    
    Formula:
        Da = SemanticDistance(user_instruction, concatenate(history_so_far))
    
    Args:
        user_instruction: The initial goal/problem description from the task
        history_so_far: List of all text strings (messages, tool calls,
                       observations) generated in the conversation up to
                       the current step
                       
    Returns:
        float: Inquiry drift score in range [0, 2] where:
               - 0.0 = conversation perfectly aligned with goal
               - ~0.5 = moderate drift
               - >1.0 = significant drift from original intent
               Returns 0.0 if calculation fails
    
    Example:
        >>> drift = calculate_inquiry_drift(
        ...     "I need to change my flight",
        ...     ["Can you help?", "Looking up flights...", "Found 3 options"]
        ... )
    """
    if not user_instruction or not history_so_far:
        return 0.0
    
    # Concatenate history into single string
    # Use newlines to separate turns for better semantic representation
    concatenated_history = "\n".join(history_so_far)
    
    if not concatenated_history.strip():
        return 0.0
    
    # Calculate semantic distance
    distance = calculate_semantic_distance(user_instruction, concatenated_history)
    
    return distance


def calculate_inference_gap(antecedent: str, consequent: str) -> float:
    """
    Calculate Inference Gap (Do) - semantic distance between action and observation.
    
    In the dual-control Tau-2 environment, this metric measures:
    - Agent Coherence: Distance between agent's tool call and its observation
    - User Coherence: Distance between agent's message and user's response
    
    Formula:
        Do = SemanticDistance(antecedent, consequent)
    
    Args:
        antecedent: What was intended/asked (agent's action or message)
        consequent: What actually happened (observation or user's response)
        
    Returns:
        float: Inference gap score in range [0, 2] where:
               - 0.0 = perfect coherence between action and outcome
               - ~0.5 = moderate gap
               - >1.0 = significant mismatch
               Returns 0.0 if calculation fails
    
    Example (Agent Coherence):
        >>> gap = calculate_inference_gap(
        ...     "Tool: get_customer_info(id='123')",
        ...     "Customer John Doe, phone: 555-1234"
        ... )
        
    Example (User Coherence):
        >>> gap = calculate_inference_gap(
        ...     "Please provide your account number",
        ...     "My number is ABC-123"
        ... )
    """
    return calculate_semantic_distance(antecedent, consequent)


# ============================================================================
# TRACER Aggregation (Final Trajectory Score)
# ============================================================================


@dataclass
class TRACERConfig:
    """
    Configuration for TRACER (Task Risk Assessment via Contextual Entropy and Reasoning) aggregation.
    
    Uses MAX variant formula (optimized via diagnostic analysis):
        risk_i = max(U_i, alpha * Da_i, beta * Do_agent_i, gamma * Do_user_i)
    
    Attributes:
        alpha: Weight for repetition (Da)
        beta: Weight for agent coherence (Do_agent) - (PRIMARY SIGNAL)
        gamma: Weight for user coherence (Do_user)
        top_k_percentile: Percentile for top-k aggregation
                         Optimized to focus on critical failure moments
                         Set to 1.0 to use all steps (equivalent to mean)
        ensemble_weight_max: Weight for single worst moment
                            TRACER = (1-w)*mean(top_k) + w*max(all_risks)
                            Combines sustained problems with single worst moment
                            Set to 0.0 to disable ensemble (use pure top-k mean)
    """
    alpha: float = 1.0
    beta: float = 1.0
    gamma: float = 1.0
    top_k_percentile: float = 0.1
    ensemble_weight_max: float = 0.1


def calculate_situational_weight(
    da: Optional[float],
    do_agent: Optional[float],
    do_user: Optional[float],
    config: TRACERConfig
) -> float:
    """
    Calculate situational penalty for a single step (additive component).
    
    The penalty combines semantic distance metrics (Da, Do) to measure
    how "risky" or "uncertain" the situational context is at step i.
    This is now used as an additive term rather than a multiplicative weight.
    
    Formula:
        Penalty_i = alpha * Da_i + beta * Do_agent_i + gamma * Do_user_i
    
    Args:
        da: Inquiry drift score (None treated as 0.0)
        do_agent: Agent coherence score (None treated as 0.0)
        do_user: User coherence score (None treated as 0.0)
        config: TRACER configuration with alpha, beta, gamma weights
        
    Returns:
        float: Situational penalty >= 0
    
    Example:
        >>> config = TRACERConfig(alpha=1.0, beta=1.0, gamma=1.0)
        >>> penalty = calculate_situational_weight(0.2, 0.3, None, config)
        >>> # Returns: 1.0*0.2 + 1.0*0.3 + 1.0*0.0 = 0.5
    """
    # Handle None values (treat as 0.0)
    da_val = da if da is not None else 0.0
    do_agent_val = do_agent if do_agent is not None else 0.0
    do_user_val = do_user if do_user is not None else 0.0
    
    # Linear combination (additive penalty)
    penalty = (
        config.alpha * da_val +
        config.beta * do_agent_val +
        config.gamma * do_user_val
    )
    
    return float(penalty)


def calculate_tracer_score(
    step_data: list[dict],
    config: Optional[TRACERConfig] = None
) -> dict:
    """
    Calculate TRACER aggregation score using MAX variant with top-k aggregation.
    
    Implements the optimized MAX variant formula:
        risk_i = max(U_i, α·Da_i, β·Do_agent_i, γ·Do_user_i)
        TRACER = (1-w) × mean(top_k%(risk_i)) + w × max(risk_i)
    
    Where:
        - risk_i takes the maximum of individual components at each step
        - top_k% = worst k% of steps (default k=15%)
        - w = ensemble weight for single worst moment (default 0.30)
        - α, β, γ = component weights (defaults: α=0.0, β=1.0, γ=0.0)
        - U_i = normalized entropy for step i
        - Da_i = inquiry drift (repetition penalty) - DISABLED by default
        - Do_agent_i = agent coherence - PRIMARY SIGNAL
        - Do_user_i = user coherence - DISABLED by default
    
    Rationale: MAX variant outperforms additive formulation (AUROC: 0.7209 vs 0.5583)
    - Takes worst signal at each step instead of summing
    - Eliminates noise from weak predictors (Da, Do_user)
    - Focuses on agent coherence (Do_agent) as primary failure indicator
    - Emphasizes critical failure moments (worst 15% + max ensemble)
    
    Args:
        step_data: List of step dictionaries containing:
                   - 'ui': normalized entropy (required)
                   - 'da': repetition score (optional)
                   - 'do_agent': agent coherence (optional)
                   - 'do_user': user coherence (optional)
        config: TRACER configuration (defaults to TRACERConfig())
        
    Returns:
        dict: {
            'tracer_score': float - final trajectory score (mean of top-k risks),
            'num_steps': int - total number of steps,
            'num_top_k': int - number of steps in top-k,
            'mean_penalty': float - average penalty across all steps,
            'std_penalty': float - std deviation of penalties,
            'mean_ui': float - average U_i across all steps,
            'penalties': list[float] - Penalty_i for each step (for debugging)
        }
    
    Example:
        >>> steps = [
        ...     {'ui': 0.1, 'da': 0.2, 'do_agent': 0.3, 'do_user': None},
        ...     {'ui': 0.15, 'da': 0.8, 'do_agent': None, 'do_user': 0.35},  # High Da
        ...     {'ui': 0.05, 'da': 0.1, 'do_agent': 0.2, 'do_user': 0.1}
        ... ]
        >>> result = calculate_tracer_score(steps)
        >>> print(f"TRACER Score: {result['tracer_score']:.4f}")
    """
    if config is None:
        config = TRACERConfig()
    
    if not step_data:
        return {
            'tracer_score': 0.0,
            'num_steps': 0,
            'num_top_k': 0,
            'mean_penalty': 0.0,
            'std_penalty': 0.0,
            'mean_ui': 0.0,
            'penalties': []
        }
    
    # Collect metrics
    penalties = []
    step_risks = []
    ui_values = []
    
    for step in step_data:
        # Extract metrics
        ui = step.get('ui', 0.0)
        da = step.get('da')
        do_agent = step.get('do_agent')
        do_user = step.get('do_user')
        
        # Handle None values (treat as 0.0)
        da_val = da if da is not None else 0.0
        do_agent_val = do_agent if do_agent is not None else 0.0
        do_user_val = do_user if do_user is not None else 0.0
        
        # Calculate penalty components
        penalty = (
            config.alpha * da_val +
            config.beta * do_agent_val +
            config.gamma * do_user_val
        )
        
        # Calculate step risk using MAX variant formula
        # risk_i = max(U_i, α·Da_i, β·Do_agent_i, γ·Do_user_i)
        step_risk = max(
            ui,
            config.alpha * da_val,
            config.beta * do_agent_val,
            config.gamma * do_user_val
        )
        
        penalties.append(penalty)
        step_risks.append(step_risk)
        ui_values.append(ui)
    
    N = len(step_data)
    
    # Top-k aggregation: take mean of highest k% of risks
    # This focuses on the worst moments which are most predictive of failure
    if config.top_k_percentile >= 1.0:
        # Use all steps (equivalent to mean)
        top_k_risks = step_risks
    else:
        # Sort risks and take top k%
        sorted_risks = sorted(step_risks, reverse=True)
        top_k_count = max(1, int(config.top_k_percentile * N))
        top_k_risks = sorted_risks[:top_k_count]
    
    # Calculate mean of top-k risks
    mean_top_k = float(np.mean(top_k_risks))
    
    # Ensemble method: combine top-k mean with max risk
    # Formula: TRACER = (1-w)*mean(top_k) + w*max(all_risks)
    # Rationale: Captures both sustained problems (mean) and single worst moment (max)
    if config.ensemble_weight_max > 0.0:
        max_risk = float(np.max(step_risks))
        tracer_score = (1 - config.ensemble_weight_max) * mean_top_k + config.ensemble_weight_max * max_risk
    else:
        # Pure top-k mean (no ensemble)
        tracer_score = mean_top_k
    
    # Calculate statistics
    result = {
        'tracer_score': tracer_score,
        'num_steps': N,
        'num_top_k': len(top_k_risks),
        'mean_penalty': float(np.mean(penalties)) if penalties else 0.0,
        'std_penalty': float(np.std(penalties)) if penalties else 0.0,
        'mean_ui': float(np.mean(ui_values)) if ui_values else 0.0,
        'penalties': penalties
    }
    
    return result


def calculate_saup_score(
    step_data: list[dict],
    config: Optional[TRACERConfig] = None
) -> dict:
    """
    Calculate SAUP (Situation-Awareness Uncertainty Propagation) score.
    
    SAUP uses a multiplicative weighting formula with RMS aggregation:
    
        SAUP = sqrt(1/N * sum_{i=1}^{N} ((W_i * U_i)^2))
    
    Where:
        - W_i = 1 + α·Da_i + β·Do_agent_i + γ·Do_user_i (penalty weight)
        - U_i = normalized entropy at step i
        - N = number of steps
    
    This differs from TRACER which uses additive combination (U_i + penalty)
    and top-k mean aggregation. SAUP uses RMS to give more weight to outlier
    moments of high weighted uncertainty.
    
    Args:
        step_data: List of dicts with keys: 'ui', 'da', 'do_agent', 'do_user'
        config: TRACER configuration for α, β, γ weights (defaults to TRACERConfig())
        
    Returns:
        Dictionary with:
            - 'saup_score': Final SAUP score (float)
            - 'num_steps': Number of steps included (int)
            - 'mean_weight': Mean penalty weight (float)
            - 'std_weight': Std dev of penalty weights (float)
            - 'mean_ui': Mean normalized entropy (float)
            - 'weights': List of W_i values (list[float])
            - 'weighted_uncertainties': List of W_i * U_i values (list[float])
            
    Example:
        >>> step_data = [
        ...     {'ui': 0.12, 'da': 0.05, 'do_agent': 0.23, 'do_user': 0.11},
        ...     {'ui': 0.15, 'da': 0.08, 'do_agent': 0.34, 'do_user': 0.09}
        ... ]
        >>> config = TRACERConfig(alpha=4.0, beta=5.0, gamma=3.0)
        >>> result = calculate_saup_score(step_data, config)
        >>> print(result['saup_score'])
        0.4523
    """
    # Initialize config to default if None
    if config is None:
        config = TRACERConfig()
    
    # Handle empty input
    if not step_data:
        return {
            'saup_score': 0.0,
            'num_steps': 0,
            'mean_weight': 0.0,
            'std_weight': 0.0,
            'mean_ui': 0.0,
            'weights': [],
            'weighted_uncertainties': []
        }
    
    # Collect metrics
    weights = []
    weighted_uncertainties = []
    ui_values = []
    
    for step in step_data:
        # Extract metrics
        ui = step.get('ui', 0.0)
        da = step.get('da')
        do_agent = step.get('do_agent')
        do_user = step.get('do_user')
        
        # Handle None values (treat as 0.0)
        da_val = da if da is not None else 0.0
        do_agent_val = do_agent if do_agent is not None else 0.0
        do_user_val = do_user if do_user is not None else 0.0
        
        # Calculate penalty: α·Da_i + β·Do_agent_i + γ·Do_user_i
        penalty = (
            config.alpha * da_val +
            config.beta * do_agent_val +
            config.gamma * do_user_val
        )
        
        # Calculate weight: W_i = 1 + penalty
        weight = 1.0 + penalty
        
        # Calculate weighted uncertainty: W_i * U_i
        weighted_uncertainty = weight * ui
        
        weights.append(weight)
        weighted_uncertainties.append(weighted_uncertainty)
        ui_values.append(ui)
    
    # Calculate SAUP using RMS (root mean square) aggregation
    # SAUP = sqrt(mean((W_i * U_i)^2))
    weighted_uncertainties_array = np.array(weighted_uncertainties)
    saup_score = float(np.sqrt(np.mean(weighted_uncertainties_array ** 2)))
    
    # Calculate statistics
    result = {
        'saup_score': saup_score,
        'num_steps': len(step_data),
        'mean_weight': float(np.mean(weights)) if weights else 0.0,
        'std_weight': float(np.std(weights)) if weights else 0.0,
        'mean_ui': float(np.mean(ui_values)) if ui_values else 0.0,
        'weights': weights,
        'weighted_uncertainties': weighted_uncertainties
    }
    
    return result


def calculate_tracer_from_trajectory(
    messages: list,
    config: Optional[TRACERConfig] = None
) -> dict:
    """
    Convenience function to calculate TRACER score directly from message objects.
    
    Extracts U_i, Da, and Do metrics from message objects and calculates
    the final TRACER trajectory score.
    
    Args:
        messages: List of message objects (AssistantMessage, UserMessage)
                  with uncertainty, da_score, do_score, do_type attributes
        config: TRACER configuration (defaults to TRACERConfig())
        
    Returns:
        dict: TRACER metrics (same as calculate_tracer_score)
        
    Example:
        >>> # After running a simulation with --calculate-uncertainty
        >>> from tau2.data_model.simulation import Results
        >>> results = Results.load("simulation.json")
        >>> sim = results.simulations[0]
        >>> tracer = calculate_tracer_from_trajectory(sim.messages)
        >>> print(f"Trajectory Score: {tracer['tracer_score']:.4f}")
    """
    step_data = []
    
    for msg in messages:
        # Only process agent and user messages
        if not hasattr(msg, 'role') or msg.role not in ['assistant', 'user']:
            continue
        
        # Skip if no uncertainty data
        if not hasattr(msg, 'uncertainty') or msg.uncertainty is None:
            continue
        
        # Extract U_i
        ui = msg.uncertainty.get('normalized_entropy', 0.0)
        
        # Extract Da
        da = msg.da_score if hasattr(msg, 'da_score') else None
        
        # Extract Do (split by type)
        do_agent = None
        do_user = None
        if hasattr(msg, 'do_score') and msg.do_score is not None:
            if hasattr(msg, 'do_type'):
                if msg.do_type == 'agent_coherence':
                    do_agent = msg.do_score
                elif msg.do_type == 'user_coherence':
                    do_user = msg.do_score
        
        step_data.append({
            'ui': ui,
            'da': da,
            'do_agent': do_agent,
            'do_user': do_user
        })
    
    return calculate_tracer_score(step_data, config)
