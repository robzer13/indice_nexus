# OroTitan — D10 : première file de traitement sur le Registry réel

**Audit SELECT uniquement, 18 octobre 2026.** Projet Supabase `cugpgtzygqqlxetyeven`. Aucune table, fonction, donnée ou politique modifiée. Aucun accès GitHub hors `robzer13/indice_nexus`. Les chiffres ci-dessous sont des observations Registry, **pas** des validations cryptographiques d'artefacts ni des recommandations d'investissement.

## Résultat vérifié

| Statut Registry | Nombre de runs |
|---|---:|
| ACTIVE | 15 |
| BLOCKED | 7 |
| PAUSED | 1 |
| READY_TO_PUBLISH | 1 (méthodologie V2.0, **pas** V3.1) |
| PUBLISHED | 34 (V1/V2 seulement) |
| **Total** | **58** |

**Répartition V3.1 exacte : cinq runs seulement**, dont deux ACTIVE / RESEARCH, deux BLOCKED / RESEARCH et un BLOCKED / DEEP_DIVE. Aucun run V3.1 en INTEGRATION ni READY_TO_PUBLISH ; ne pas présenter le writer D7 comme prouvé de bout en bout sur des cas V3.1.

## File V3.1 à traiter

| Run UUID | Situation source | Blocage ou priorité observée |
|---|---|---|
| `48b9426b-4d0d-4858-9d87-efcd13ac415d` | ACTIVE / RESEARCH ; CHECKPOINT / gate NO | `BLOCKED_INSUFFICIENT_INPUT` : contradictions d'adoption clients, qualification indépendante HB, courbe technologique, ROIC et cash owner insuffisamment réconciliés ; approfondir Research, ne pas forcer Deep Dive |
| `5d6d7a93-828e-476a-93a3-37f4ac7a2ff9` | BLOCKED / RESEARCH ; CHECKPOINT / gate NO | `BLOCKED_INSUFFICIENT_INPUT` : rapport audité FY2025/26 non entièrement lisible, retours sur acquisitions non bornés, réfutation moat et dénominateur de marché manquants, share count/treasury/options non clos, owner earnings incomplets |
| `98383baf-fc0d-4ba8-8d0b-18d7b7d3dbe0` | BLOCKED / RESEARCH ; CHECKPOINT / gate NO | `BLOCKED_INSUFFICIENT_INPUT` : share count au 12 octobre UNKNOWN, conflit net debt et preuves indépendantes de rendements/concessions de trésorerie insuffisantes |
| `d56a0bc9-4e80-48b3-b326-74d1fea15e63` | ACTIVE / RESEARCH ; sans manifeste actif | Recherche en cours ; aucune preuve de progression admissible via un CHECKPOINT actif, continuer et vérifier la disponibilité d'un manifeste avant handoff |
| `a4cf9002-b52d-4440-8902-adc70bd777dd` | BLOCKED / DEEP_DIVE ; CHECKPOINT | `VALUATION_BLOCKED_ECONOMIC_SHARE_COUNT_UNRESOLVED` : dénominateur exact ou borné non prouvé au 22 septembre 2026 ; interdiction de transporter le count du 16 septembre sans preuve de tous les mouvements entre dates |

**Recommandation opérationnelle :** avancer d'abord les preuves d'inputs Research et la réconciliation des mouvements d'actions, plutôt que d'ouvrir un nouveau lot de contrats de publication. Le run `d56a0bc9...` peut être suivi dans l'interface sans le déclarer admissible ; les autres nécessitent des sources concrètes avant reprise.

## Limite révélée par la base et par D5

Les artefacts ont le backend `PRIVATE_GITHUB` ; nous n'avons pas consulté le dépôt d'artefacts externe. Le type exact `SCORING_INPUTS_FULL_PRECISION` attendu par D5 n'est enregistré sur aucun run, alors que le Registry contient notamment `VALUATION_FULL_PRECISION_INPUTS` ou `SCORING_INPUTS_OUTPUTS` dans des générations précédentes. Cette divergence de typage **bloquerait** D5 pour une future Integration FINAL V3.1 tant qu'une autorité de mapping de contenu et de filiation n'est pas approuvée. Ne pas relabeliser les pièces ni créer de fausse attestation.

## Produit livré dans la PR #383, non déployé

Le cockpit `/admin/orotitan` montre désormais une **file V3.1 par défaut**, le nombre de runs actifs/bloqués, des filtres `V3.1 à traiter`, `Tous les blocages` et `Toutes les versions`, le code métier de l'étape, la synthèse Registry et, si fourni, `required_resolution` ou `resolution_required`. Le contrôle D2–D8 **n'est appelé que pour les vrais runs process ET Pilotage V3.1**, jamais pour V1/V2/V3.0. Tout run déclaré BLOCKED/PAUSED/CANCELLED est présenté en diagnostic sans admission.

Les textes de la colonne Registry restent des **observations non vérifiées** ; seule la décision du chargeur D2–D8, si ses preuves et son observation indépendante d'environnement sont disponibles, bénéficie du label de vérification. Aucun bouton de publication, aucune mutation, aucune migration ni déploiement.

## Contrôles read-only reproductibles

```sql
SELECT process_version, pilotage_contract_version, run_status, current_stage, count(*)
FROM public.orotitan_runs GROUP BY 1,2,3,4 ORDER BY 1,2,3,4;

SELECT r.run_id,r.run_status,r.current_stage,r.process_version,
       s.lifecycle_status,s.contract_status_code,s.blocker_summary
FROM public.orotitan_runs r
JOIN public.orotitan_run_stages s
  ON s.run_id=r.run_id AND s.stage_code=r.current_stage
WHERE r.process_version='3.1' AND r.pilotage_contract_version='3.1';
```

Le droit d'ouvrir un run V3.1 ne vaut pas droit de modifier sa stage, d'accéder à un dépôt externe ou de publier un snapshot. Pour déployer cette interface, une approbation séparée de la PR et de la configuration runtime reste nécessaire.
