"""Generate actionable coaching recommendations."""
from __future__ import annotations

from typing import Dict, Iterable, List

from .schemas import BiasSignal

_RECOMMENDATIONS: Dict[str, List[str]] = {
    "confirmation": [
        "Planifier une revue hebdomadaire des scénarios alternatifs.",
        "Consigner deux signaux contradictoires avant toute entrée.",
    ],
    "recency": [
        "Comparer chaque trade aux statistiques 6-12 mois pour relativiser l'évènement récent.",
    ],
    "overconfidence": [
        "Limiter la taille maximale d'une nouvelle position à 60% de la taille moyenne gagnante.",
        "Valider systématiquement le plan avec un pair ou un checklist objective.",
    ],
    "loss_aversion": [
        "Automatiser les stops à -2R et auditer leur respect chaque semaine.",
        "Programmer une alerte lorsque la durée de détention dépasse la moyenne historique des gagnants.",
    ],
    "fomo": [
        "Mettre en pause 15 minutes avant toute entrée post-news et vérifier le ratio risque/rendement.",
    ],
    "disposition": [
        "Fixer des objectifs partiels et un plan de suivi pour les positions gagnantes.",
    ],
    "familiarity": [
        "Introduire au moins une nouvelle valeur ou secteur par mois dans la watchlist.",
    ],
    "sunk_cost": [
        "Limiter à une seule moyenne à la baisse par idée et documenter la justification.",
    ],
    "herding": [
        "Évaluer l'indépendance du signal via trois sources différentes avant de suivre le consensus.",
    ],
}


def build_recommendations(biases: Iterable[BiasSignal], limit: int = 5) -> List[str]:
    """Return up to ``limit`` unique coaching recommendations."""

    suggestions: List[str] = []
    for bias in biases:
        for message in _RECOMMENDATIONS.get(bias.bias_id, []):
            if message not in suggestions:
                suggestions.append(message)
            if len(suggestions) >= limit:
                break
        if len(suggestions) >= limit:
            break
    if not suggestions:
        suggestions.append("Maintenir la discipline actuelle et continuer le journal de trading.")
    return suggestions


__all__ = ["build_recommendations"]
