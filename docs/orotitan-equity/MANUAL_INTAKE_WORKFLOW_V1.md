# OroTitan — retour manuel de livrables ChatGPT (implémentation pilote)

**Statut : en PR, non activé en production.** Pas de modification méthodologique, pas de writer Registry.

## Parcours sur PC
1. Ouvrir un run dans `/admin/orotitan?run=<RUN_ID>`.
2. Consulter la barre des cinq phases et les blockers.
3. Copier et exécuter manuellement le prompt dans ChatGPT.
4. Télécharger les livrables produits (pour ce premier lot : .json, .md, .txt, .csv).
5. Dans le bloc « Déposer les livrables ChatGPT », sélectionner **un** fichier (UTF-8, non vide, ≤ 1 Mo), puis « Déposer sans valider l’étape ».
6. Voir le statut **Reçu · non vérifié** et l’empreinte SHA-256 du contenu reçu.

## Autorité et séparation
- Les fichiers résident dans le bucket privé existant `orotitan-text-artifacts-v1`,
  **uniquement** sous `manual-intake/v1/<RUN_UUID>/<CURRENT_REGISTRY_STAGE>/<SHA256>__<FILENAME>`.
- Les octets sont vérifiés uniquement au niveau *format* (UTF-8, JSON objet, SHA local côté serveur).
  Cela ne constitue **pas** une attestation de stockage indépendante, une revue de sources,
  une admissibilité de calcul, une signature d'auteur, une autorité d'artefact ni une
  certitude sur le contenu analytique.
- Cette interface **ne crée aucun** `orotitan_artifacts`, `PERSISTENCE_ATTESTED`,
  stage checkpoint/finalization, `READY_FOR_*`, `GO PUBLISH`, snapshot ni appel IA.
- L'obtention des artefacts officiels et l'admission du prochain prompt restent soumises
  au circuit exact déjà autorisé (persisted bytes, lineage, contrôle de contrat,
  gates, Registry et vérification indépendante). Aucun `start_orotitan_stage` n'est appelé.
- Le chemin de stockage isolé ne doit **jamais** être interprété par le trusted loader comme
  preuve autoritative. Les objets non validés ne peuvent pas être promus par simple renommage.
- Le bucket est privé ; la route admin seule utilise la clé serveur, jamais exposée au navigateur.
- Dépôt explicitement désactivé tant que `OROTITAN_MANUAL_INTAKE_ENABLED` n'est pas la chaîne
  `true` dans la configuration serveur. Ne pas activer sans validation du déploiement/écriture
  de stockage de production.

## Contrôles
- Session admin serveur obligatoire en GET/POST, URL Supabase épinglée, run préexistant,
  stage contrôlé par la base et non le formulaire, statut autorisé ACTIVE / PAUSED / BLOCKED.
- POST exige une origine strictement identique au site courant et multipart/form-data.
- Un fichier par demande, format restrictif, taille maximale 1 Mo, encodage UTF-8,
  JSON racine objet, max. 100 objets par étape, écriture sans upsert.
- En cas de défaillance de listing/upload : erreur, aucune affirmation de réception.
  Si un objet exact existe déjà, la réponse précise le caractère doublon sans proclamer sa vérification.
- Aucune suppression/édition de fichiers dans cette version. La gestion de rétention,
  la promotion par vérificateur et l'admission des artefacts sont des jalons distincts.

## Limites du pilote
- Dépôt manuel des textes et tableaux, pas encore de PDF, DOCX, images ou documents lourds.
- La liste indique l'état de réception dans le stockage privé ; **elle ne certifie pas**
  l'état des artefacts du Registry. Les étapes et leurs gates restent visibles
  dans le cockpit principal, séparément.
- Pour Besi, Research CHECKPOINT/NO reste bloqué : le dépôt ne le transforme pas en FINAL.
