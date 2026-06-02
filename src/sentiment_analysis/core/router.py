import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

class SemanticRouter:
    """
    Implements SynaptoRoute for Semantic Caching and Intent Triage.
    This bypasses the heavy Transformer/LLM model for non-emotional queries.
    """
    def __init__(self):
        self.router: Any = None
        self.encoder: Any = None
        self.storage: Any = None
        self._initialize_router()

    def _initialize_router(self):
        try:
            from synaptoroute import AdaptiveRouter, Route
            from synaptoroute.encoder import FastEmbedEncoder
            from synaptoroute.profile import ProfileType, get_profile
            from synaptoroute.storage import SQLiteStorage

            # Use SQLite to persist routes across restarts
            db_path = Path("src/sentiment_analysis/data/routes.db")
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # Apply Latency Profile
            profile = get_profile(ProfileType.LATENCY)

            # Note: profile.threads must be passed explicitly to FastEmbedEncoder
            self.encoder = FastEmbedEncoder(threads=profile.threads)
            self.storage = SQLiteStorage(str(db_path))
            self.router = AdaptiveRouter(self.encoder, self.storage, profile=profile)

            self._setup_triage_routes(Route)
            logger.info("SynaptoRoute initialized successfully.")

        except ImportError:
            logger.warning("SynaptoRoute not installed. Semantic triage is disabled.")

    def _setup_triage_routes(self, route_cls: Any) -> None:
        """Creates predefined routes for non-emotional intent triage."""
        if self.router is None:
            return

        # Check if the route exists to avoid duplicate work
        try:
            existing = self.router("what time is it")
            if existing and existing.name == "non_emotional_factual":
                return # Already set up
        except Exception:
            pass

        factual_route = route_cls(
            name="non_emotional_factual",
            utterances=[
                "what time is it",
                "how tall is the eiffel tower",
                "what is the capital of france",
                "who is the president",
                "calculate 5 plus 10",
                "what is the weather today",
                "define the word dictionary",
                "how many miles to the moon",
            ],
            threshold=0.85 # High threshold so we don't accidentally block sarcasm like "brilliant deduction"
        )

        greeting_route = route_cls(
            name="non_emotional_greeting",
            utterances=[
                "hello",
                "hi there",
                "good morning",
                "how are you",
                "sup",
                "hey",
            ],
            threshold=0.85
        )

        self.router.add_route(factual_route)
        self.router.add_route(greeting_route)

    def triage_intent(self, text: str) -> str | None:
        """
        Routes the text to determine if it can bypass the heavy sentiment model.
        Returns the route name if matched (e.g., 'non_emotional_factual'), else None.
        """
        if not self.router:
            return None

        # SynaptoRoute processes the query
        result = self.router(text)

        if result and result.name:
            logger.info(f"SynaptoRoute tripped! Intercepted query as: {result.name}")
            return result.name

        return None

# Singleton instance
semantic_router = SemanticRouter()
