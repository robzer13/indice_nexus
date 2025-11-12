"""Definitions for behavioural biases monitored by OroTitan."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True, slots=True)
class BiasDefinition:
    """Declarative description of a behavioural bias."""

    bias_id: str
    label: str
    description: str
    weight: float


_BIAS_DEFINITIONS: List[BiasDefinition] = [
    BiasDefinition(
        bias_id="confirmation",
        label="Confirmation Bias",
        description="Tendance à ne rechercher que les informations allant dans le sens du plan initial.",
        weight=0.12,
    ),
    BiasDefinition(
        bias_id="recency",
        label="Recency Bias",
        description="Poids excessif donné aux événements les plus récents.",
        weight=0.08,
    ),
    BiasDefinition(
        bias_id="overconfidence",
        label="Overconfidence",
        description="Surestimation de la capacité à prédire ou contrôler le marché.",
        weight=0.1,
    ),
    BiasDefinition(
        bias_id="loss_aversion",
        label="Loss Aversion",
        description="Propension à laisser courir les pertes tout en coupant rapidement les gains.",
        weight=0.12,
    ),
    BiasDefinition(
        bias_id="fomo",
        label="FOMO",
        description="Peurs de manquer un mouvement entraînant des entrées tardives.",
        weight=0.08,
    ),
    BiasDefinition(
        bias_id="disposition",
        label="Disposition Effect",
        description="Vendre trop tôt les gagnants et conserver les perdants.",
        weight=0.1,
    ),
    BiasDefinition(
        bias_id="familiarity",
        label="Familiarity",
        description="Sur-exposition à des actifs déjà connus.",
        weight=0.07,
    ),
    BiasDefinition(
        bias_id="sunk_cost",
        label="Sunk Cost",
        description="Renforcement systématique des positions perdantes.",
        weight=0.11,
    ),
    BiasDefinition(
        bias_id="herding",
        label="Herding",
        description="Suivi du consensus de marché sans validation propre.",
        weight=0.07,
    ),
]


def get_bias_definitions() -> List[BiasDefinition]:
    """Return all supported bias definitions."""

    return list(_BIAS_DEFINITIONS)


def iter_biases() -> Iterable[BiasDefinition]:
    """Iterate over the configured bias definitions."""

    return iter(_BIAS_DEFINITIONS)


__all__ = ["BiasDefinition", "get_bias_definitions", "iter_biases"]
