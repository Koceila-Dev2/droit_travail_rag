# Compte-Rendu : RAG appliqué au Code du Travail

Ce document résume les défis techniques rencontrés lors de l'intégration du RAG (Retrieval-Augmented Generation) sur le texte juridique du Code du Travail, ainsi que les solutions de conception adoptées pour la viabilité de l'architecture.

---

## 1. Choix de conception (Design Choices)

- **Du Local vers l'API Cloud (Embedding) :**
Initialement pensés pour un modèle `SentenceTransformers` tournant en local (`distiluse-base-multilingual-cased-v2`), nous nous sommes vite heurtés à la fenêtre de contexte maximale très stricte de 128 tokens. Le Code du Travail regorgeant d'articles-fleuves, la sémantique originelle était perdue ou complètement tronquée. Nous avons donc pivoté vers l'API de Mistral AI (`mistral-embed`) qui offre à la fois un nombre impressionnant de tokens maximum par morceau (8192) et une excellente compréhension de la syntaxe et des spécificités de la langue française.

- **Stratégie de "Chunking" Mixée :**
Puisque le droit se lit en contexte, la solution optimale était de faire du *chunking hiérarchique*, en associant à chaque morceau de l'Article les informations structurelles (`Partie` > `Livre` > `Titre`). Toutefois, pour respecter strictement la limite de l'API de Mistral, une couche de *Sliding Window Chunking* (Texte par chevauchement) a été injectée. La taille du chunk cible est plafonnée à 2500 mots, chevauchant les 300 prochains mots du texte pour ne briser aucune disposition légale.

- **Mécanismes de Cache :**
Générer et télécharger un embedding à chaque test pour expérimenter différents modèles mathématiques / vecteurs mettait en péril les temps d'exécution et les tarifs d'utilisation de l'API. Nous avons implémenté un système de sauvegarde "hors ligne" (Pickle local `embeddings_response_mistral.pckl`) pour court-circuiter le réseau de manière transparente si un premier appel avait déjà eu lieu.

---

## 2. Difficultés rencontrées et Fixes déployés

### A. Les articles "éléphantesques"
L'algorithme de séparation initial n'opérait qu'une stricte séparation par Titre/Article. C'est l'API de Mistral AI qui, par une `SDKError 400`, nous a mis sur la piste qu'un simple article (comme le fameux "Article R4314-17") faisait exploser la limite maximale de tokenisation du service de par ses 6000 mots (~11400 tokens perçus en Mistral-Token).
**Solution :** Refacto total du script `chunk_generator.py` pour y inclure un `split_text_with_overlap`. L'article R4314-17 est désormais découpé proprement en 5 sous-chapitres (via `total_chunks: 5`).

### B. Limites d'insertion (Batching API et DB)
Bien que les limites individuelles (par chunk) aient été réglées, deux services interféraient avec la quantité insérée **en même temps** :
1. **Mistral AI :** Refusait un batch trop lourd ("Too many tokens overall"). La procédure d'appel massif ("Fire And Forget") a été déconstruite pour être rezippée en mini-paquets successifs (tous les `batch_size = 40`).
2. **ChromaDB :** L'instance `PersistentClient` crachait une `ValueError` au-dessus de 5461 documents transférés simultanément en mémoire vectorielle.
**Solution :** L'encapsulation via une nouvelle boucle modulaire limitant le batch `chroma_batch_size` à 5000 ids par itération.

### C. Problème d'encodage de l'Environnement Windows
Pendant le packaging du projet (`pip freeze`), un classique de PowerShell a corrompu le rendu textuel UTF de `requirements.txt`, rendant la mise sous dépôt ou test tiers impossible (caractères `NUL` interpolés en `UTF-16 LE`).
**Solution :** Utilisation directe du Pipe Powershell avec forçage de l'encodage via `Out-File -Encoding utf8`.