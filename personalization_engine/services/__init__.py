"""
services package — the business-logic layer of the Personalization Engine.

Re-exports let other layers write:
    from services import PersonalizationService, DriverMemoryService

Three services, one per concern:
    PersonalizationService -> WHO is driving + comfort/UI preferences (original)
    DriverMemoryService    -> Learning Loop, state machine, advisory thresholds
    ExplanationService     -> grounded Vietnamese explanations (GetExplanation)
"""

from services.personalization_service import (
    DriverNotFoundError,
    PersonalizationService,
    ServiceError,
)
from services.memory_service import (
    DriverMemoryService,
    MemoryServiceError,
    TripNotFoundError,
)
from services.explanation_service import (
    EventNotFoundError,
    ExplanationService,
    ExplanationServiceError,
)

__all__ = [
    # original driver identification service
    "DriverNotFoundError",
    "PersonalizationService",
    "ServiceError",
    # driver memory / learning loop
    "DriverMemoryService",
    "MemoryServiceError",
    "TripNotFoundError",
    # explanations
    "EventNotFoundError",
    "ExplanationService",
    "ExplanationServiceError",
]
