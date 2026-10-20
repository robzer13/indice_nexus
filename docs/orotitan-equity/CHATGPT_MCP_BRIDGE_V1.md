# OroTitan ↔ ChatGPT — pont MCP Option A (prototype, non déployé)

## Intention produit
OroTitan est le cockpit (étapes, preuve, prochaines actions). ChatGPT est le lieu de travail
**manuel**. Le serveur MCP distant permet de rappeler les informations du run et,
si toutes les vérifications sont admises, de recevoir le prompt exact sans recopie.
La transmission de fichiers textuels peut devenir un outil d'écriture **distinct,
explicitement activé** vers le stockage privé de quarantaine ; elle ne clôt aucune étape.

## Contrat des outils MCP
Sur l'endpoint `/api/orotitan/mcp` (Streamable HTTP, SDK officiel MCP v2) :
- `get_orotitan_run({run_id})` — identité du dossier, étape, code de blocage, manifest,
  CAS : **observations Registry non certifiées** ; le nom/ticker sont des étiquettes.
- `get_orotitan_next_prompt({run_id})` — vérification D2 de l'environnement et de la
  provenance/bytes immuables sur GitHub, admission D1/Registry sur versions CAS récentes ;
  retourne le prompt exact uniquement quand `CONTINUE` ou `HANDOFF` est admis.
  En cas de blocage et/ou de projet non observé : **aucun prompt de l'étape suivante**.
- `list_orotitan_submissions({run_id})` — reçus des fichiers sous le chemin
  `manual-intake/v1/<RUN>/<REGISTRY_STAGE>`; état uniquement
  `RECEIVED_NOT_VERIFIED` ; ne prétend jamais à une attestation d'artefact.
- `submit_orotitan_work_product(...)` — **non annoncé par défaut** ;
  le texte soumis est borné à 100 000 caractères et aux types .json objet,
  .md, .txt, .csv, UTF-8, 1 Mo selon le validateur existant. Demande
  `expected_stage` et `expected_run_state_version` lus dans le Registry.
  Écrit sans upsert exclusivement dans le namespace de quarantaine,
  jamais directement dans `orotitan_artifacts` ou une étape.

Aucun outil d'analyse IA, de promotion d'artefacts GitHub, de modification
du Registry, de finalisation, de publication ou de modification de schéma.

## Authentification et phase pilote (NON PRÊT À CONNECTER EN PRODUCTION)

L'endpoint est **coupé par défaut** :
`OROTITAN_MCP_ENABLED=false`. Il exige en plus :
- `OROTITAN_MCP_PUBLIC_ORIGIN` : origine HTTPS exacte de l'application,
  sans route ni barre oblique terminale ;
- `OROTITAN_MCP_READ_TOKEN` : secret serveur d'au moins 48 caractères,
  présenté uniquement dans l'en-tête HTTP Authorization Bearer ;
- `OROTITAN_MCP_ALLOWED_RUN_IDS` : allow-list de 1 à 25 UUID existants exacts.
  Aucune route ne peut explorer un run non autorisé.

Le dépôt MCP est doublement désactivé :
`OROTITAN_MCP_WRITE_ENABLED=false` et
`OROTITAN_MANUAL_INTAKE_ENABLED=false`. La correspondance des versions est revérifiée
au dernier moment, l'écriture restant une quarantaine et non un stage writer.

**Limitation réelle :** le portail de connexion des plugins ChatGPT ne propose pas
actuellement un flux d'installation par API key/bearer statique ; sa connexion
personnalisée requiert une méthode de connexion et les droits du compte
compatibles. Ce prototype au token fixe est un **point de départ pour des tests
HTTP privés**, pas une connexion ChatGPT déjà installable. Avant toute activation
publique : intégrer **OAuth 2.1 / PKCE, scopes read/write, rotation/révocation
de tokens**, gestion d'identité distincte de la session admin navigateur,
protection contre abus, déploiement HTTPS isolé, puis tests avec ChatGPT.
La prise en charge des actions d'écriture peut dépendre de l'offre ChatGPT ;
ne promettre aucune disponibilité sans vérification du compte.

## Test de non-régression et gates
- Les tests unitaires définissent les protections : rejet sans token, rejet
  d'UUID non whitelistes, double flag pour écriture, route SDK officiel,
  réutilisation D1/D2, absence de mutation du Registry.
- Aucun appel Supabase/GitHub de production n'est nécessaire aux tests.
- Un résultat `RECEIVED_NOT_VERIFIED` n'est pas un manifest FINAL.
- Un Research CHECKPOINT/NO (notamment le run Besi bloqué) ne doit jamais
  être promu à Fundamentals sur la base de son simple enregistrement.
- Les contrats gelés, les preuves GitHub et le projet Supabase `orotitan-screener`
  restent les autorités existantes ; `vnext-shadow` est exclu.

## Jalons de mise en service séparés
1. Validation du code PR / test HTTP isolé (route désactivée par défaut).
2. Ajouter OAuth et tester l'authentification dans ChatGPT selon les droits
   réellement disponibles.
3. Déploiement de la route lecture seule, strictement bornée aux runs autorisés,
   après autorisation explicite.
4. Préparation puis autorisation *distincte* d'une transmission de livrables en
   quarantaine, avec confirmation utilisateur.
5. Certification/Registry writer séparé. Ne jamais considérer la réception
   ou la copie du prompt comme une validation analytique.

## Références externes
- https://developers.openai.com/plugins/build/mcp-server
- https://developers.openai.com/plugins/build/auth
- https://developers.openai.com/plugins/deploy/submission
- https://help.openai.com/en/articles/12584461-connecting-custom-mcp-servers-in-chatgpt
- https://github.com/modelcontextprotocol/typescript-sdk/blob/main/docs/serving/http.md
