# Code du Travail - RAG (Retrieval-Augmented Generation)

Ce projet implémente un système de Question-Réponse (RAG) spécialisé sur le Code du travail français. Il s'appuie sur une base de données vectorielle locale et l'API de Mistral AI pour générer des réponses précises et contextualisées.

## Choix Techniques

- **Modèle d'Embeddings :** `mistral-embed` de Mistral AI. Bien supérieur aux petits modèles locaux (comme `distiluse-base-multilingual-cased-v2` qui plafonnait à 128 tokens) pour comprendre la sémantique complexe du droit français, avec une large fenêtre de tokens (8192).
- **Base de Données Vectorielle :** ChromaDB. C'est une solution robuste, locale et persistante (`PersistentClient`) parfaite pour un RAG léger sans déployer d'infrastructure lourde.
- **Stratégie de Chunking :** Extraction basée sur la structure du droit (Partie > Livre > Titre > Article) couplée à un découpage fenêtré (overlap de 300 mots) limitant la taille à 2500 mots par chunk, afin de ne jamais dépasser la fenêtre de tokens de Mistral tout en empêchant la perte de contexte.
- **Stockage local des APIs:** Mise en cache des appels d'embeddings dans un fichier `embeddings_response_mistral.pckl` pour économiser les quotas de l'API et accélérer les tests.

## Comment lancer le projet

### 1. Prérequis et Installation
- Créez et activez un environnement virtuel (si ce n'est pas déjà fait) :
  ```powershell
  python -m venv venv
  .\venv\Scripts\Activate.ps1
  ```
- Installez les librairies du wrapper Mistral et ChromaDB :
  ```powershell
  pip install -r requirements.txt
  ```

### 2. Configuration API
Ajoutez un fichier `.env` à la racine avec votre jeton :
```env
MISTRAL_API_KEY=votre_cle_api_ici
GROQ_API_KEY=votre_cle_api_ici
```

### 3. Exécution du Pipeline

**A. Découpage du document source :**
```powershell
python chunk_generator.py
```
*(Analyse les articles et divise de façon adaptative dans `chuncks.pckl`)*

**B. Embedding et Indexation :**
```powershell
python vector_db.py
```
*(Interroge l'API Mistral par lot, génère `droits_travail_embeddings_db` avec ChromaDB, et gère le cache)*

**C. Inférence (Lancement du RAG) :**
```powershell
python rag.py
```
*(Extrait le contexte pertinent par rapport à la question et génère la réponse)*