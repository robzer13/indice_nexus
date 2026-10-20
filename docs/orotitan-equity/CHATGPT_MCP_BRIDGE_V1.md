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

## Authentification OAuth 2.1 préparée (sans émission de jetons)

**Le connecteur n'est pas déployé ni raccordé à ChatGPT.**

Le bridge fonctionne en mode `OAUTH` seulement lorsque les paramètres
serveur ci-dessous sont configurés, et `OROTITAN_MCP_ENABLED=true`.
Un fournisseur d'identité externe **établi** doit délivrer des access tokens
JWT RS256. OroTitan est uniquement le **resource server** : il ne possède
ni endpoint d'autorisation, ni secret client OAuth, ni user store séparé.

Variables d'environnement (jamais `NEXT_PUBLIC_`) :
- `OROTITAN_MCP_PUBLIC_ORIGIN` : origine HTTPS exacte du déploiement approuvé ;
- `OROTITAN_MCP_AUTH_MODE=OAUTH` ;
- `OROTITAN_MCP_OAUTH_ISSUER` : issuer OAuth HTTPS exact ;
- `OROTITAN_MCP_OAUTH_JWKS_URI` : JWKS HTTPS sur le même hôte que l'issuer ;
- `OROTITAN_MCP_OAUTH_ALLOWED_SUBJECTS` : allow-list des identifiants OAuth `sub` ;
- `OROTITAN_MCP_ALLOWED_RUN_IDS` : allow-list de 1 à 25 UUID du Registry.

Chaque requête MCP valide la **signature RS256 via les clés JWKS externes**
(`jose`), l'issuer `iss`, l'audience `aud` strictement égale à
`https://<app>/api/orotitan/mcp`, l'expiration, l'âge du token
(≤ 1 heure), l'identité `sub` dans l'allow-list et la portée
`orotitan.read`. Un token avec `orotitan.write` seulement ne suffit pas.
Une écriture exige en plus `orotitan.write` dans le jeton et les
deux flags indépendants `OROTITAN_MCP_WRITE_ENABLED=true` et
`OROTITAN_MANUAL_INTAKE_ENABLED=true`. Les lots JSON-RPC sont
refusés lorsque l'écriture est activée, pour ne pas contourner les scopes.

Découverte RFC 9728 :
`/.well-known/oauth-protected-resource` n'est servie que lorsque
l'auth OAuth est configurée ; les 401 du MCP indiquent cette URL via
`WWW-Authenticate`. Cela **ne remplace pas** le serveur d'autorisation
du fournisseur d'identité, lequel doit exposer sa discovery OAuth et
assurer l'autorization-code + PKCE S256, la liaison `resource` → audience,
l'identification du client par CIMD/DCR, le consentement et le renouvellement
des access tokens. Les protocoles de révocation et de renouvellement
restent la responsabilité du fournisseur choisi et de son exploitation.

**Limite du SDK actuellement épinglé :** `@modelcontextprotocol/server@2.0.0`
ne permet pas de déclarer directement `securitySchemes` sur
`registerTool` dans ses types. Les scopes sont donc appliqués
strictement à la frontière HTTP ; la prise en charge complète de
l'interface d'authentification par outil et sa métadonnée nécessite
une mise à jour du SDK ou un adaptateur testé. Aucune compatibilité
d'installation ChatGPT n'est revendiquée tant que ce point n'est pas
validé de bout en bout.

Le raccourci hérité par bearer statique ne fonctionne qu'avec
`OROTITAN_MCP_AUTH_MODE=TEST_BEARER` ET `NODE_ENV != production`.
Il ne donne aucun moyen de contourner OAuth en production.

La documentation du Help Center en date du 20 octobre 2026
annonce les connexions MCP personnalisées pour **ChatGPT Business
et Enterprise/Edu** avec permissions adéquates. Aucun droit
d'installation ne peut être déduit du seul code du serveur.
https://help.openai.com/en/articles/12584461-connecting-custom-mcp-servers-in-chatgpt

## Test de non-régression et gates
- Les tests unitaires définissent les protections : rejet sans token, rejet
  d'UUID non whitelistes, double flag pour écriture, route SDK officiel,
  réutilisation D1/D2, absence de mutation du Registry.
- Les tests de signatures JWT/claims RS256 utilisent des clés éphémères locales,
  sans fournisseur externe ni jeton véritable.
- Une lecture Supabase de Besi a été réalisée séparément en lecture seule :
  état ACTIVE/RESEARCH, version run 6, CHECKPOINT/NO, six blocages. Ce constat
  ne démontre pas un échange MCP réussi depuis ChatGPT.
- Aucun appel Supabase/GitHub de production n'est nécessaire aux tests de code.
- Un résultat `RECEIVED_NOT_VERIFIED` n'est pas un manifest FINAL.
- Un Research CHECKPOINT/NO (notamment le run Besi bloqué) ne doit jamais
  être promu à Fundamentals sur la base de son simple enregistrement.
- Les contrats gelés, les preuves GitHub et le projet Supabase `orotitan-screener`
  restent les autorités existantes ; `vnext-shadow` est exclu.

## Jalons de mise en service séparés
1. Validation du code PR / test HTTP isolé (route désactivée par défaut).
2. Configurer un IdP OAuth 2.1 compatible (PKCE/CIMD ou DCR/resource/audience),
   valider la métadonnée d'auth par outil, puis tester ChatGPT avec un compte
   et les droits réels nécessaires.
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


## D15 — Consentement Supabase Auth : maquette isolée et prérequis effectifs

Une page `/oauth/consent?authorization_id=...` a été ajoutée à la PR,
**OFF par défaut** avec `OROTITAN_OAUTH_CONSENT_ENABLED=false`.
Son fonctionnement prévu respecte le parcours décrit dans la documentation
Supabase OAuth Server (version bêta) :

- Connexion explicite avec un **véritable utilisateur Supabase Auth déjà
  créé** (par mot de passe, sans inscription automatique), distinct du
  mot de passe administrateur OroTitan.
- Vérification de session Supabase par `auth.getUser()`, puis
  contrôle serveur via `auth.getUser(access_token)` et liste d'UUID
  utilisateurs autorisés; aucune confiance aveugle dans le stockage navigateur.
- Lecture de `authorization_id` via
  `supabase.auth.oauth.getAuthorizationDetails`, affiche le client,
  l'URL de retour et les scopes effectivement demandés.
- Consentement **explicite** : `approveAuthorization` ou
  `denyAuthorization`. Uniquement les callbacks connus de
  `chatgpt.com` sont admis, avec préservation du code et du state.
- L'écran ne confère lui-même aucun droit de lecture du Registry :
  le serveur MCP doit toujours vérifier le JWT OAuth, le bon run
  et l'admission du prompt.
- Routes `/oauth/consent` et `/api/orotitan/oauth/eligibility`
  non opérationnelles hors déploiement HTTPS/configuration expresse.
  La connexion admin HMAC existante demeure intacte.

**Blocage réellement constaté au 2026-10-20, lecture seule :
`auth.users` dans `orotitan-screener` contient zéro utilisateur.**
Avant toute activation, un administrateur doit créer/inviter
une identité Supabase Auth humaine destinée à cette connexion
et recueillir son consentement. Ne jamais transmettre un mot de
passe, une clé OAuth ou un jeton dans une conversation ChatGPT.

**Points de compatibilité encore ouverts, non contournés :**

1. Le `Site URL` Auth affiché dans Supabase est
   `http://localhost:3000` ; `/oauth/consent` doit être réellement
   accessible sous un domaine HTTPS revu avant activation. Changer le
   Site URL peut affecter les flux d'authentification existants.
2. La signature par défaut des JWT Supabase peut être `HS256`, tandis
   que le resource server MCP n'accepte actuellement que `RS256`
   et une JWKS HTTPS. Vérifier l'algorithme du projet et planifier
   une migration asymétrique séparément si nécessaire.
3. Les scopes standards Supabase OAuth sont `openid`, `email`,
   `profile`, `phone`. Notre MCP requiert les claims
   `orotitan.read` et une audience `/api/orotitan/mcp` exacte.
   **Aucun jeton Supabase standard ne doit être supposé admissible.**
   Il faut tester/implémenter les claims adaptés via la fonctionnalité
   `Custom Access Token Hook`, sans écraser les claims essentiels
   Supabase, et certifier l'échange avant déploiement.
4. ChatGPT exige OAuth 2.1 / PKCE S256, une discovery d'AS
   cohérente, et identification du client CIMD, DCR ou client
   pré-enregistré. Les callback URLs doivent être exactes. Le
   serveur MCP doit aussi fournir la métadonnée d'outil compatible.
5. L'environnement Vercel de prévisualisation est protégé par SSO;
   prévoir un accès de test explicitement autorisé plutôt que
   désactiver globalement la protection.
6. Les paramètres d'activation OAuth et les secrets sont tous
   **absents** de la configuration Vercel consultée. Pas de
   configuration automatique ni de merge/deploy/activation.

**Conditions de sortie pour un premier test Besi :** un utilisateur Auth
admis, un issuer/JWKS vérifié, token scoped vers l'audience MCP,
un endpoint joignable depuis ChatGPT, la connexion OAuth de ChatGPT
réussie, et une lecture indépendante du run
`48b9426b-4d0d-4858-9d87-efcd13ac415d`
confirmant Research CHECKPOINT/NO et ses six blockers.
Aucun progrès de phase n'est déduit de la connexion.

Références :
- https://supabase.com/docs/guides/auth/oauth-server/getting-started
- https://supabase.com/docs/guides/auth/oauth-server/mcp-authentication
- https://developers.openai.com/plugins/build/auth
