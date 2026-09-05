# OroTitan Screener V1.1

OroTitan Screener V1.1 transforme la V1 validée en cockpit opérationnel quotidien. La source de vérité financière reste inchangée : les analyses sont des snapshots immutables et les cours sont append-only.

## Objectif

Répondre rapidement à quatre questions :

1. Quelles sociétés sont à O90 ou proches de O90 ?
2. Quelles données doivent être actualisées ?
3. Comment ajouter une société ou une analyse sans casser l’historique ?
4. Les cours automatiques fonctionnent-ils correctement ?

## Nouveautés V1.1

### Radar d’entrée

Chaque société est classée dans une zone déterministe à partir de la distance O90 :

- O90 atteint : distance >= 0 %
- à moins de 5 % : -5 % <= distance < 0 %
- à 5–10 % : -10 % <= distance < -5 %
- à 10–20 % : -20 % <= distance < -10 %
- à plus de 20 % : distance < -20 %
- Non calibré : distance NULL

La formule O90 reste strictement :

```text
(price_o90 / current_price - 1) * 100
```

### Screener avancé

`/screener` ajoute :

- pays ;
- secteur ;
- score minimum ;
- distance minimum / maximum ;
- zone d’entrée ;
- calibré / non calibré ;
- fraîcheur du cours ;
- tri secondaire.

Les NULL restent hors tri numérique.

### Admin restructuré

Routes V1.1 :

- `/admin` : cockpit d’administration ;
- `/admin/companies` : référentiel des sociétés ;
- `/admin/companies/new` : création ;
- `/admin/companies/[slug]` : modification des métadonnées ;
- `/admin/snapshots/new` : snapshot structuré ou import JSON ;
- `/admin/prices` : rafraîchissement manuel et journal des synchronisations ;
- `/admin/data-health` : contrôle des données.

Aucune route admin ne permet de supprimer un snapshot ou un cours.

### Import JSON

Le formulaire rapide accepte un objet JSON analytique avec, au minimum :

```json
{
  "analysis_date": "2026-09-05",
  "model_version": "OROTITAN-DEEP-2026-09",
  "status": "FINALIST",
  "source_title": "Nom du rapport"
}
```

Les champs absents qui sont facultatifs deviennent NULL. `score_components` reste un objet JSON libre.

### Gestion des sociétés

Les métadonnées suivantes deviennent administrables :

- slug ;
- ticker ;
- nom ;
- marché ;
- devise ;
- unité MAJOR/MINOR ;
- décimales ;
- symbole Twelve Data ;
- multiplicateur de marché ;
- pays ;
- secteur ;
- active/inactive.

Cette modification ne touche jamais les snapshots ni les prix historiques.

### Cours

Le moteur de prix est partagé entre le cron et le bouton admin.

Chaque exécution V1.1 écrit un journal append-only dans `market_sync_runs` avec :

- déclencheur CRON ou ADMIN ;
- début / fin ;
- nombre de sociétés ;
- insertions ;
- échecs ;
- résultat détaillé par société.

### Data Health

Le contrôle détecte notamment :

- cours manquant ;
- cours périmé ;
- O90 absent ;
- symbole marché absent ;
- multiplicateur invalide ;
- analyse absente ;
- analyse de plus de 180 jours ;
- pays ou secteur manquant.

Ce module ne remplace jamais une donnée manquante.

### Fiche société

La fiche ajoute :

- zone d’entrée ;
- historique des prix ;
- lignes de seuil O85/O90/O92/O95 ;
- comparaison du snapshot courant au précédent sur score, fair value centrale et O90.

Le graphique n’utilise que `market_prices`. Les seuils proviennent uniquement du snapshot analytique courant.

## Migration depuis V1

La base Supabase V1 existante doit recevoir une seule migration avant d’utiliser les fonctions d’écriture V1.1.

Dans Supabase SQL Editor, exécuter :

```text
migrations/v1_1.sql
```

Cette migration :

- crée `market_sync_runs` ;
- rend ce journal append-only ;
- ajoute le trigger `updated_at` des sociétés ;
- autorise `service_role` à INSERT/UPDATE sur `companies` ;
- n’accorde aucun DELETE ;
- ne modifie aucun snapshot ni prix existant.

Pour une installation neuve, `schema.sql` contient déjà la structure V1.1 complète.

## Sécurité

- Supabase reste server-only.
- `SUPABASE_SERVICE_ROLE_KEY` ne doit jamais être publique.
- RLS reste activé.
- Les rôles `anon` et `authenticated` n’ont pas d’accès direct aux tables.
- `snapshots` reste protégé par un trigger UPDATE/DELETE.
- `market_sync_runs` est append-only.
- `companies` peut être INSERT/UPDATE mais pas DELETE.
- La session admin utilise un cookie HTTP-only signé HMAC.
- `ADMIN_SESSION_SECRET` reste recommandé. Si Vercel ne l’injecte pas, V1.1 dérive côté serveur une clé HMAC depuis `ADMIN_PASSWORD`.

## Variables

```dotenv
NEXT_PUBLIC_SUPABASE_URL=https://YOUR_PROJECT.supabase.co
SUPABASE_SERVICE_ROLE_KEY=...
ADMIN_PASSWORD=...
ADMIN_SESSION_SECRET=...
TWELVE_DATA_API_KEY=...
CRON_SECRET=...
```

## Déploiement

1. Déployer la branche V1.1 en Preview Vercel.
2. Exécuter `migrations/v1_1.sql` sur Supabase.
3. Vérifier :
   - `/`
   - `/screener`
   - une fiche société
   - `/admin`
   - `/admin/companies`
   - `/admin/snapshots/new`
   - `/admin/prices`
   - `/admin/data-health`
4. Lancer une actualisation manuelle des prix.
5. Contrôler le journal.
6. Fusionner dans `main` uniquement après validation.

## Vérification locale / CI

```bash
npm run lint
npm run typecheck
npm test
npm run build
```

Les tests couvrent désormais la V1 et la V1.1 : formule O90, NULL, formatage devises, Auto Trader en pence, priorisation, immutabilité, zones d’entrée, validation de société, Data Health et migration SQL.

## Hors périmètre V1.1

Toujours hors périmètre :

- portefeuille et sizing ;
- trading/exécution ;
- découverte automatique de sociétés ;
- notifications ;
- DCF interactif ;
- analyse technique ;
- génération IA ;
- intégration NEXUS complète ;
- graphiques avancés multi-modèles.

Ces sujets relèvent d’une future V2.
