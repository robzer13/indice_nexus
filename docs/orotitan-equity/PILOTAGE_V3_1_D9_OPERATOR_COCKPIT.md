# OroTitan D9 — Cockpit administrateur opérationnel (lecture seule)

## Produit livré dans la PR #383 (non déployée)

**URL de l'application après fusion et déploiement approuvés séparément : `/admin/orotitan`.**

Le cockpit permet de sélectionner un run enregistré dans `public.orotitan_runs` (les 30 plus récents ou un UUID saisi), de consulter son statut réel, son étape courante, son `data_cutoff` et sa version de CAS observée. Cette **observation brute du Registry n'est pas une attestation**.

Une vérification approfondie distincte appelle le chargeur serveur D2–D8 (manifeste exact, artefacts immuables, SHA-256, Git blob, verrous métier, cohérence V3.1, recalcul I2, provenance D6 et précontrôle historique). La décision affichée indique sa phase, son motif, ses versions run/étape, son empreinte d'état et le handoff proposé, strictement **non exécuté**. Toute erreur donne un code de blocage et aucun faux statut de réussite.

Le routeur est un composant Next.js serveur. L'administration est protégée par la session signée `orotitan_admin_session`. Aucune route d'écriture, aucun RPC, aucune invocation de writer, aucune autorisation GO PUBLISH, aucun endpoint de preview public n'ont été ajoutés. `publicationAuthorized=false` et `mutationPerformed=false` à chaque étape. Même un `AWAIT_EXPLICIT_GO_PUBLISH` de l'adaptateur est transformé en état bloqué dans la projection opérateur tant que le writer V3.1 est interdit.

## Vérification indépendante de l'environnement

Le serveur n'invente pas un environnement `ACTIVE_HEALTHY`. Pour vérifier l'étape sélectionnée, il demande à l'API Management Supabase `GET https://api.supabase.com/v1/projects/<project_ref>` une observation indépendante, puis rapproche **projet, nom, région et statut** des pins de runtime V3.1.

La variable **serveur uniquement** `SUPABASE_MANAGEMENT_READ_TOKEN` est facultative pour la liste et indispensable pour une décision cryptographiquement vérifiée. Utiliser un PAT Supabase **limité au projet** `cugpgtzygqqlxetyeven`, permission **Project Settings / Read** ; ne jamais le préfixer par `NEXT_PUBLIC_`, ne pas le commiter. La lecture de la liste nécessite seulement les variables Supabase serveur et la session admin existantes.

Le chargeur D2 des objets immuables conserve séparément les exigences existantes `OROTITAN_PRIVATE_GITHUB_REPOSITORIES` et `OROTITAN_PRIVATE_GITHUB_TOKEN`, gérées à l'exécution. **Aucun autre dépôt GitHub n'a été consulté ou modifié lors de D9.** Si les secrets de lecture des artefacts ne sont pas configurés, le diagnostic approfondi échoue normalement fermé.

## Validation produit

- Route `/admin/orotitan` ajoutée à `components/admin/admin-nav.tsx`.
- Projections opérateur testables dans `pilotage-operator-view-v3-1.ts` ; aucun faux GO PUBLISH.
- Chargement sécurisé dans `pilotage-operator-cockpit-server.ts` ; validation du host Supabase avant tout accès, contrôle indépendant du projet Management, snapshots Registry lus deux fois par D2 ; aucune mutation.
- Tests de présentation et de sécurité dans `tests/equity-v3-pilotage-operator-cockpit.test.ts`.
- Schémas AJV auparavant lus par `readFileSync(new URL(...))` au chargement des modules maintenant importés statiquement en JSON, pour assurer le build Next.js. Les contrats JSON, règles d'analyse et permissions n'ont pas été modifiés.
- Référence D8 technique mise à jour en `OROTITAN_D8_REVIEW_BASELINE_V1.1` avec l'empreinte de code révisée. Le gel méthodologique et la publication restent **NO GO**.

## Constat sur les données réelles, le 18/10/2026

Une requête SELECT isolée sur le projet de production a compté **58 runs**, dont **15 ACTIVE**, avec **8 RESEARCH, 14 DEEP_DIVE et 36 INTEGRATION** tous statuts confondus. Ce comptage ne constitue pas un run smoke-test D2–D8 ; les résultats affichés par la future UI n'ont pas encore été observés sur un déploiement réel.

## Limite opérationnelle

Le code doit être revu, fusionné et déployé par décision séparée pour que `/admin/orotitan` soit réellement accessible à un administrateur. D9 produit **un point d'entrée produit concret et testable**, mais ne constitue ni un writer transactionnel V3.1 ni la levée des quatre blocages d'homologation. Ne pas confondre la réussite du build avec l'accès effectif à l'URL en production.

**Statut publication : NO GO.** Aucune migration, aucune création de snapshot ni transition canonique n'est autorisée.
