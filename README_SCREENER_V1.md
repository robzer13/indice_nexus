# OroTitan Screener V1

> Module web du dépôt `indice_nexus`. Le moteur Python historique reste conservé séparément dans ce même repository.

# OroTitan Screener V1

OroTitan Screener est le cockpit d'une base fermée de sociétés déjà analysées. Il répond à une question unique : quelles sociétés suivies sont aujourd'hui les plus proches d'un point d'entrée OroTitan ?

La V1 conserve trois entités : `companies`, `snapshots` et `market_prices`. Les analyses sont versionnées et immutables. Les cours sont append-only et ne réécrivent jamais une analyse.

## Stack

- Next.js 16 App Router, React 19, TypeScript strict
- React Server Components par défaut
- Tailwind CSS
- Supabase Postgres
- Vercel et Vercel Cron
- Twelve Data pour les prix quotidiens

Les seuls composants client concernent l'interactivité du screener. Supabase n'est appelé que côté serveur avec la service role key.

## Routes

- `/` : KPIs, fraîcheur, six priorités dérivées des données
- `/screener` : recherche, filtres et tris sur un dataset agrégé côté serveur
- `/company/[slug]` : identité, état, valorisation, analyse, score components et historique
- `/admin` : authentification personnelle et création append-only de snapshots
- `/api/cron/prices` : ingestion Twelve Data protégée par `CRON_SECRET`

## Installation

Prérequis : Node.js 22 et npm.

```bash
npm install
cp .env.example .env.local
npm run dev
```

Ouvrir ensuite `http://localhost:3000`.

## Supabase

1. Créer un projet Supabase.
2. Dans le SQL Editor, exécuter `schema.sql`.
3. Exécuter ensuite `seed.sql`.
4. Renseigner l'URL du projet et la service role key dans `.env.local` et Vercel.

Le schéma active RLS sur les trois tables, ne crée aucune policy publique, retire les privilèges directs aux rôles navigateur et réserve les opérations applicatives à `service_role`. La vue `latest_company_state` sélectionne le dernier cours et le dernier snapshot analytique sans fusionner plusieurs modèles.

### Immutabilité

La clé métier d'un snapshot est :

```text
company_id + analysis_date + model_version
```

Le bootstrap source utilisait initialement `ON CONFLICT (...) DO UPDATE` sur les snapshots. La V1 le corrige en :

```sql
on conflict (company_id,analysis_date,model_version) do nothing;
```

Le schéma ajoute en outre un trigger `BEFORE UPDATE OR DELETE` sur `snapshots`. Même une écriture via service role doit créer une nouvelle date ou `model_version` plutôt que modifier un snapshot existant. Les prix bootstrap utilisent `WHERE NOT EXISTS` afin d'éviter le doublon exact sans supprimer ni mettre à jour l'historique.

## Dataset bootstrap

Les données bootstrap sont conservées pour Hermès, RATIONAL, Scout24, Auto Trader, SEI Investments, Baltic Classifieds, Qualys et Medistim. Aucun nombre manquant n'est inventé : un `NULL` reste `NULL`.

## Variables d'environnement

```dotenv
NEXT_PUBLIC_SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
ADMIN_PASSWORD=...
ADMIN_SESSION_SECRET=...
TWELVE_DATA_API_KEY=...
CRON_SECRET=...
```

`SUPABASE_SERVICE_ROLE_KEY` ne doit jamais utiliser le préfixe `NEXT_PUBLIC_`. `ADMIN_SESSION_SECRET` doit contenir au moins 32 caractères aléatoires. Aucun secret réel ne doit être committé.

## Admin

`/admin` vérifie `ADMIN_PASSWORD` côté serveur. Après authentification, la session est stockée dans un cookie signé HMAC, HTTP-only, `SameSite=Lax`, `Secure` en production, avec une durée de huit heures.

Toutes les écritures revérifient la session côté serveur. L'admin ne propose ni UPDATE ni DELETE. Si la combinaison métier existe déjà, l'insertion est refusée explicitement. `source_title` est obligatoire. Les champs vides deviennent `NULL`, jamais zéro.

## Distance O90

Formule unique :

```text
distance_o90_pct = (price_o90 / current_price - 1) * 100
```

- négatif : le cours doit encore baisser ;
- zéro : cours au seuil O90 ;
- positif : cours déjà inférieur au seuil ;
- O90 absent ou donnée invalide : `null`, affiché « Non calibré ».

Le dashboard ne classe que des distances calculables. Le screener trie les nombres normalement et place les `NULL` après les valeurs numériques.

## Devises, unités et Auto Trader

Le helper central `formatPrice` utilise `currency`, `quote_unit` et `price_decimals`.

Auto Trader reste exprimée en pence : `currency=GBP`, `quote_unit=MINOR`, `price_decimals=0`, `market_data_multiplier=100`. Une valeur métier `528` s'affiche donc `528p`, sans conversion arbitraire en livres.

## Twelve Data et Vercel Cron

`GET /api/cron/prices` :

1. exige `Authorization: Bearer <CRON_SECRET>` ;
2. charge les sociétés actives ayant un `market_data_symbol` ;
3. appelle Twelve Data `/price` avec timeout ;
4. valide un prix strictement positif ;
5. applique `market_data_multiplier` ;
6. insère une nouvelle ligne dans `market_prices` ;
7. isole les erreurs par société ;
8. renvoie un JSON explicite `inserted/failed`.

La clé Twelve Data n'est jamais envoyée au navigateur.

`vercel.json` programme `/api/cron/prices` à 22:15 UTC du lundi au vendredi.

## Fraîcheur

La V1 signale un cours comme « Données anciennes » après 96 heures sans nouveau point. La fiche société affiche aussi l'horodatage et la source du cours.

## Déploiement Vercel

1. Importer le repository dans Vercel.
2. Utiliser le preset Next.js.
3. Ajouter les six variables d'environnement.
4. Déployer.
5. Vérifier `/`, `/screener`, une fiche société et `/admin`.
6. Déclencher le cron avec le bearer secret pour valider l'intégration Twelve Data.

Les pages de données sont dynamiques : le build n'a pas besoin de secrets Supabase présents pour compiler.

## Vérification

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

Les tests couvrent la distance O90, les `NULL`, le formatage EUR/USD/GBP minor/NOK, la priorisation, la garde d'immutabilité applicative, l'unicité et le trigger SQL, ainsi que le `DO NOTHING` du seed.

## Limites V1

Sont hors périmètre : news, DCF interactif, analyse technique, portefeuille, sizing, trading/exécution, recherche automatique de nouvelles sociétés, génération IA, moteur NEXUS, notifications et comparaison graphique avancée multi-modèle.

L'historique multi-modèle est conservé et affiché sans fusion silencieuse.
