"""Literature grounding: OpenAlex, Crossref, web search, review communities."""

from open_referee.literature.context import ContextPack, LiteratureItem
from open_referee.literature.crossref import CrossrefClient
from open_referee.literature.openalex_client import OpenAlexClient
from open_referee.literature.scout import PeerReviewScout
from open_referee.literature.search import SearchHit, SearchRouter

__all__ = [
    "ContextPack",
    "LiteratureItem",
    "OpenAlexClient",
    "CrossrefClient",
    "SearchRouter",
    "SearchHit",
    "PeerReviewScout",
]
