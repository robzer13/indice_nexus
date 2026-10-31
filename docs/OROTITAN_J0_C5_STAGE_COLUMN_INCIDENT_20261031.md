# J0/C5 — 2026-10-31 correction de la projection Registry

## Incident observé
L'exécution du contrôle HTTP C5 affichait `NON VÉRIFIÉ — J0_C5_RUN_INVARIANT_FAILED`.
L'échec intervenait **avant** l'exécution des quatre requêtes HTTP de sécurité.

## Diagnostic confirmé sur SHADOW
- La requête PostgREST `orotitan_run_stages.select('stage_id',{count:'exact',head:true})` ciblait une colonne **inexistante**.
- Dans le Registry canonique, `orotitan_run_stages` contient notamment `run_id`, `stage_code`, `stage_revision` ; il ne contient pas `stage_id`.
- L'orchestration regroupait les erreurs de lecture de colonnes et les véritables anomalies du run sous `J0_C5_RUN_INVARIANT_FAILED` ; cela masquait la cause.
- Le run L’Oréal `204390a1-5cba-4b9c-9821-ca0c7df43eca` était toujours `CREATED`, `state_version=2`, `current_stage=NULL`, sans stage, artefact ou snapshot.

## Correction
- Lecture `select('run_id',{count:'exact',head:true})` pour dénombrer les stages.
- Erreurs distinctes pour les lectures `RUN`, `STAGE`, `ARTIFACT`, `SNAPSHOT` et les vrais écarts d'état.
- Régression `tests/runfactory-j0-c5-registry-column-contract-v1.test.ts` assurant l'absence de `stage_id` sur ce chemin.

## Conditions de sortie
Cette correction est limitée au diagnostic **SHADOW**. Elle ne modifie pas le run, la production, le journal J0, les permissions, la protection Vercel, les contrats figés ni les critères d’acceptation.
Ne pas clôturer `issue:j0-c5-preview-http-sso-20261031` avant quatre réponses HTTP applicatives vérifiées, reçu persistant et relecture indépendante.
