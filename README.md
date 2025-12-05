# Architecture et Implémentation de la Classe ChessTRM : Une Approche Récursive pour la Modélisation Échiquéenne

## 1. Introduction : Le Changement de Paradigme vers la Récursion Compacte

L'industrie des moteurs d'échecs neuronaux a longtemps été dominée par une corrélation directe entre la profondeur architecturale (le nombre de couches physiques) et la force de jeu. Des architectures comme AlphaZero ou Leela Chess Zero (Lc0) reposent sur des réseaux résiduels (ResNets) massifs, empilant des dizaines, voire des centaines de blocs de convolution pour extraire des caractéristiques de plus en plus abstraites. Cependant, une nouvelle frontière de recherche, matérialisée par le Tiny Recursive Model (TRM), remet en question ce dogme en proposant que la profondeur de raisonnement ne nécessite pas une profondeur de paramètres correspondante.

Votre projet, qui dispose désormais d'un fichier H5 prêt à l'emploi, se situe à l'avant-garde de cette transition. L'objectif n'est plus de construire un réseau statique profond, mais d'implémenter un moteur dynamique, la classe ChessTRM, capable de réutiliser cycliquement un noyau neuronal minimaliste (souvent 2 couches seulement) pour raffiner itérativement sa compréhension de l'échiquier. Ce rapport détaille l'ingénierie nécessaire pour traduire les données brutes de votre fichier H5 en une intelligence récursive fonctionnelle, en mettant l'accent sur l'implémentation de la classe, la gestion des flux de données ($x, y, z$) et l'application critique de la Supervision d'Amélioration Profonde (Deep Improvement Supervision - DIS).

### 1.1 La Philosophie "Less is More" appliquée aux Échecs

L'architecture TRM repose sur l'hypothèse que la complexité computationnelle doit être déployée dans le temps (itérations) plutôt que dans l'espace (paramètres). Alors que les grands modèles de langage (LLM) ou les moteurs d'échecs classiques effectuent une passe avant unique (ou séquentielle fixe) pour générer une réponse, le TRM fonctionne comme une boucle de rétroaction cognitive. Il maintient une "pensée" courante ($z$) et une "ébauche de réponse" ($y$), qu'il critique et améliore à chaque cycle.

Dans le contexte échiquéen, cela mime le processus de réflexion humaine : un Grand Maître ne perçoit pas instantanément la vérité de la position. Il formule une hypothèse candidate (le coup $y_0$), évalue ses conséquences tactiques via une simulation mentale (mise à jour de $z$), et révise son jugement pour produire $y_1$, puis $y_2$, convergeant vers le coup optimal. L'implémentation de la classe ChessTRM doit capturer cette dynamique temporelle, transformant votre fichier H5 statique en une séquence d'apprentissage dynamique.

### 1.2 Aperçu des Composants Critiques

Pour réussir cette implémentation, nous devons orchestrer trois composants techniques majeurs qui structureront ce rapport :

* **L'Interface d'Entrée-Sortie :** La traduction des tenseurs H5 (19 plans) vers l'espace latent du modèle.
* **Le Noyau Récursif (The Recursive Core) :** L'architecture du Transformer partagé qui fusionne la perception ($x$), la mémoire ($z$) et l'intention ($y$).
* **Le Mécanisme de Supervision (DIS) :** La méthodologie d'entraînement spécifique qui force le modèle à s'améliorer à chaque étape, rendant la récursion stable et productive.

## 2. Architecture des Données et Spécifications des Tenseurs

L'efficacité du TRM dépend intrinsèquement de la qualité et de la structure des données qu'il ingère. Contrairement aux réseaux convolutifs qui tolèrent une grande redondance spatiale, les modèles basés sur l'Attention (comme le TRM) nécessitent une représentation dense et sémantiquement riche. Votre fichier H5 doit être mappé vers des structures de données précises au sein de la classe ChessTRM.

### 2.1 Représentation de l'État ($x$ - Input Stream)

La littérature technique sur AlphaZero et ses dérivés établit un standard pour la représentation de l'échiquier. Bien que Lc0 utilise jusqu'à 112 plans pour capturer des historiques profonds, une architecture "Tiny" commande une approche plus parcimonieuse pour éviter de diluer la capacité du modèle dans des embeddings d'entrée trop vastes. Nous recommandons une structure en 19 plans, alignée sur les implémentations fondamentales d'AlphaZero.

L'implémentation de la classe doit prévoir une couche de projection initiale (InputEmbedding) capable de traiter un tenseur de forme adaptée.

| Index du Plan | Description Sémantique | Type de Donnée | Rôle dans le TRM |
| :--- | :--- | :--- | :--- |
| 0 - 5 | Pièces du Joueur Actif (P, N, B, R, Q, K) | Binaire (0/1) | Tactique Immédiate : Définit les menaces et les ressources matérielles directes. |
| 6 - 11 | Pièces de l'Adversaire (P, N, B, R, Q, K) | Binaire (0/1) | Contrainte : Définit les obstacles et les cibles pour le calcul du flux $z$. |
| 12 - 13 | Répétitions (1 fois, 2 fois) | Binaire (0/1) | Contexte Stratégique : Crucial pour éviter les nuls involontaires, information stockée dans $z$. |
| 14 - 17 | Droits de Roque (Blanc O-O, O-O-O, Noir...) | Constant (sur tout le plan) | Planification : Influence la sécurité du roi à long terme. |
| 18 | Trait (Couleur au trait) | Constant | Orientation : Permet au modèle d'inverser la perspective si nécessaire (bien que l'entrée soit souvent relative). |

**Analyse d'Implémentation :**
Le tenseur $x$ ne change pas au cours de la récursion. Il sert d'ancrage ("anchor") à la réalité. Dans la méthode forward, ce tenseur $x$ sera projeté une seule fois vers la dimension du modèle ($d_{model}$), puis réinjecté à chaque pas de temps $t$. Cela diffère des RNN classiques où l'entrée change à chaque pas ; ici, l'entrée est stationnaire, c'est le traitement qui est dynamique.

### 2.2 Représentation de l'Action ($y$ - Answer Stream)

Le flux $y$ représente la "réponse" du modèle. Aux échecs, cela correspond à une distribution de probabilité sur l'espace des coups légaux. L'espace d'action standard UCI (Universal Chess Interface) comprend environ 1968 coups possibles (couvrant tous les déplacements "from-to" et les promotions).

Contrairement aux architectures classiques qui produisent $y$ uniquement à la toute fin, le TRM considère $y$ comme une entrée récurrente.

* **Initialisation ($y_0$) :** Au début de la réflexion ($t=0$), le modèle ne sait rien. $y_0$ est généralement initialisé comme un embedding apprenable statique (un "token de départ") ou un vecteur de bruit gaussien de faible amplitude ($\sigma=0.02$) pour briser la symétrie.
* **Projection Inverse :** La classe doit inclure une tête de lecture (ReadoutHead) qui projette l'état latent $y_t$ (dimension $d_{model}$) vers un vecteur de taille 1968 (logits).
* **Réinjection :** Pour l'étape suivante $t+1$, ces logits (ou leur version softmaxée) doivent être re-projetés vers $d_{model}$.

**Insight Technique :** La dimensionnalité de 1968 est élevée par rapport à la dimension interne typique d'un "Tiny" modèle (ex: $d_{model}=256$ ou $512$). L'utilisation d'une factorisation de matrice ou d'un embedding dense pour les coups est essentielle pour ne pas exploser le nombre de paramètres dans la couche de sortie.

### 2.3 Gestion de l'Espace : Aplanissement vs Patching

Le TRM étant souvent basé sur des Transformers, il attend une séquence de tokens en entrée. L'échiquier est une grille 2D $8 \times 8$. Deux stratégies s'affrontent dans la littérature :

* **Patching (Vision Transformers) :** Découper l'image en patchs (ex: $4 \times 4$). Pour un échiquier $8 \times 8$, cela ferait trop peu de tokens (4 tokens de $4 \times 4$) pour capturer la complexité tactique fine.
* **Aplanissement (Flattening) :** Traiter chaque case comme un token unique. Cela génère une séquence de longueur 64 ($8 \times 8 = 64$).

**Recommandation pour ChessTRM :** L'approche par Aplanissement est impérative. Chaque case a une sémantique propre (occupée ou vide, blanche ou noire).

* **Séquence :** 64 tokens.
* **Encodage Positionnel :** Puisque le Transformer est invariant par permutation, vous devez ajouter des Positional Embeddings. Pour les échecs, des embeddings positionnels 2D (séparant rangées et colonnes) ou 1D absolus (0 à 63) sont nécessaires pour que le modèle "comprenne" les diagonales et les mouvements de cavaliers.

## 3. Architecture Interne de la Classe ChessTRM

Cette section détaille le cœur de l'implémentation. La classe ChessTRM hérite typiquement de `torch.nn.Module`. Elle ne contient pas une "profondeur" de couches, mais un bloc unique réutilisé.

### 3.1 Initialisation et Hyperparamètres (`__init__`)

La définition des hyperparamètres doit balancer la contrainte "Tiny" avec la complexité des échecs. Les recherches sur le TRM suggèrent qu'un modèle de 7 millions de paramètres peut surpasser des modèles beaucoup plus grands grâce à la récursion.

Paramètres suggérés pour l'initialisation :
* `d_model` : 256 ou 512 (Largeur du vecteur latent pour chaque case).
* `n_heads` : 8 (Nombre de têtes d'attention).
* `n_layers` : 2 (Nombre de couches Transformer dans le bloc récursif. C'est très peu, mais c'est le principe du TRM).
* `dropout` : 0.1 (Essentiel pour la régularisation dans les boucles récurrentes).
* `max_recursion` ($T$) : 8 à 16 (Nombre maximal d'itérations en inférence).

### 3.2 Le Mécanisme de Fusion des Flux ($x, y, z$)

Le point critique de l'implémentation est la méthode `combine_streams`. Comment $x$ (l'échiquier), $y$ (l'idée de coup actuelle) et $z$ (la réflexion abstraite) interagissent-ils?

Les snippets de code de recherche indiquent une approche par addition ou concaténation avant l'entrée dans le bloc Transformer. Pour les échecs, l'addition élément par élément (element-wise addition) est souvent préférée pour conserver la correspondance spatiale stricte : la case E4 du flux $x$ doit s'additionner à la case E4 du flux $z$.

$$Input_{t} = \text{Embed}(x) + \text{Proj}(y_{t-1}) + z_{t-1}$$

Cependant, il existe une nuance importante : $y$ (le coup global) n'est pas spatialement isomorphe à l'échiquier. $y$ est un vecteur de distribution de coups (1968 dims).

**Insight d'Implémentation :** Il faut "diffuser" (broadcast) l'information de $y$ sur les 64 tokens de l'échiquier, ou concaténer un token "Global $y$" à la séquence de 64 tokens.

Une approche plus élégante pour le TRM est de traiter $y$ comme un token spécial (comme le CLS token dans BERT) ou de projeter $y$ vers les dimensions spatiales si l'on utilise une représentation de type "From-Square" et "To-Square".

**Recommandation Simplifiée :** Considérez $y$ comme un vecteur global concaténé au début de la séquence (longueur $64 + 1$). Le mécanisme d'attention permettra aux cases de "lire" l'intention globale $y$, et à $y$ de "lire" l'état des cases.

### 3.3 Le Bloc Récursif (Transformer Block)

Le bloc répété doit être standard mais robuste.

**Pre-Norm vs Post-Norm :** Dans les réseaux récurrents profonds (ce que devient le TRM une fois déroulé sur 16 pas), la Pre-Normalization (LayerNorm appliquée avant l'attention et le FFN) est cruciale pour la stabilité des gradients.

**Structure :**
1. LayerNorm
2. Multi-Head Self Attention (permet aux pièces d'interagir globalement).
3. Add Residual
4. LayerNorm
5. FeedForward Network (MLP avec activation GeLU ou SwiGLU).
6. Add Residual

### 3.4 La Boucle forward et la Gestion d'État

L'implémentation de la méthode forward n'est pas triviale. Elle doit gérer le "déroulement" (unrolling) de la boucle pour l'entraînement.

```python
def forward(self, x, n_steps=None):
    # Encodage initial
    x_emb = self.input_projection(x) + self.pos_encoding
    batch_size = x.shape[0]
    
    # Initialisation des états latents
    # z est le "scratchpad" mental
    z = torch.randn_like(x_emb) * 0.02 
    # y est l'intention de coup (token global ou distribué)
    y_emb = self.y_init.expand(batch_size, -1, -1) 
    
    all_outputs = []
    
    # Boucle de Récursion (Déroulement temporel)
    steps = n_steps if n_steps else self.default_steps
    for t in range(steps):
        # 1. Fusion des flux
        # Le modèle "voit" le plateau (x), sa pensée précédente (z) 
        # et son intention précédente (y)
        combined_input = self.combine(x_emb, y_emb, z)
        
        # 2. Passage dans le Noyau (Shared Weights)
        # Le noyau met à jour la pensée z
        z_new = self.transformer_block(combined_input)
        
        # 3. Mise à jour de l'intention y
        # Souvent, une partie légère du réseau ou le même bloc
        # est utilisé pour extraire la nouvelle intention à partir de z_new
        y_emb_new = self.y_update_layer(z_new)
        
        # 4. Connexions Résiduelles Temporelles (Crucial pour DIS)
        # On n'écrase pas totalement z, on l'affine
        z = z + self.gating(z_new) # Gating optionnel
        y_emb = y_emb_new
        
        # 5. Décodage pour supervision
        # On produit une prédiction "réelle" à chaque pas pour la DIS
        logits = self.policy_head(y_emb)
        all_outputs.append(logits)
        
    return all_outputs # Retourne l'historique complet pour la Loss DIS
```

## 4. Supervision d'Amélioration Profonde (DIS) : Le Moteur d'Apprentissage

C'est ici que l'implémentation se distingue d'un simple RNN. La Deep Improvement Supervision (DIS) est la méthodologie d'entraînement qui permet à un si petit modèle de converger. Sans DIS, le signal de gradient traversant 16 couches récurrentes s'évanouirait ou exploserait, et le modèle peinerait à lier l'entrée $x$ à la sortie $y_{final}$.

### 4.1 La Fonction de Perte DIS (Loss Function)

Au lieu de calculer la perte uniquement sur la dernière sortie ($y_{final}$), la DIS calcule une perte à chaque étape intermédiaire $t$.

$$\mathcal{L}_{DIS} = \sum_{t=1}^{T} w_t \cdot \mathcal{L}_{CE}(\text{Softmax}(y_t), \text{Target})$$

* $\mathcal{L}_{CE}$ : Cross-Entropy Loss standard.
* $y_t$ : Logits prédits à l'étape $t$.
* Target : Le coup cible (vérité terrain issue du fichier H5).

**Analyse des Cibles Intermédiaires :**
Certaines variantes de DIS suggèrent d'utiliser des "cibles diffusées" (plus floues au début, plus nettes à la fin). Pour les échecs, modifier la cible est risqué (un coup "presque" légal n'a pas de sens). Il est préférable de garder la même cible dure (le meilleur coup) pour toutes les étapes, mais de laisser le modèle converger vers une distribution de probabilité "Peaked" (pointue). Le modèle apprendra naturellement à avoir une entropie élevée (incertitude) aux étapes $t=1, 2$ et une entropie faible à $t=T$.

### 4.2 Le Scheduling des Poids ($w_t$)

La pondération des étapes $w_t$ est critique. Si $w_1$ est trop fort, le modèle essaie de "deviner" immédiatement sans réfléchir, court-circuitant le processus récursif (effet "Greedy"). Si $w_T$ est trop faible, il ne raffine pas assez.

Deux stratégies de scheduling sont identifiées dans les recherches :

| Stratégie | Description | Poids (t=1→T) | Avantage |
| :--- | :--- | :--- | :--- |
| **Linéaire Croissante** | Augmente l'importance de la précision à mesure que le temps avance. | $[0.1, 0.2, 0.4, 0.8, 1.0]$ | Favorise l'exploration au début (raisonnement $z$) et la précision à la fin. **Recommandé.** |
| **Uniforme** | Tous les pas ont la même importance. | $[1.0, 1.0, 1.0, 1.0, 1.0]$ | Force chaque étape à être une approximation valide. Utile pour l'Adaptive Computation Time (arrêt précoce). |
| **Curriculum Decay** | Décroissance exponentielle (inspirée de CGAR). | $[1.0, 0.8, 0.6,...]$ | Contre-intuitif pour les échecs, plus adapté aux tâches de débruitage pur. |

**Implémentation recommandée :** Utilisez une pondération linéaire croissante. $w_t = \frac{t}{T}$. Cela dit au modèle : "Je ne t'en veux pas si tu te trompes au début, mais sois précis à la fin".

### 4.3 Simplification de l'Adaptive Computational Time (ACT)

Le modèle HRM (prédécesseur du TRM) utilisait un mécanisme complexe d'arrêt (ACT) avec une tête de "Halting" et une pénalité de temps. Le TRM simplifie cela drastiquement.

Dans votre implémentation ChessTRM, n'incluez pas de tête de Halting explicite pour la première version. La DIS suffit à structurer le calcul. À l'inférence, vous fixerez simplement un nombre de pas $N$ (ex: 10) ou arrêterez quand la distribution de probabilité $y_t$ se stabilise (Delta entre $y_t$ et $y_{t-1}$ inférieur à un seuil $\epsilon$).

## 5. Implémentation Logicielle et Optimisation

Cette section aborde les aspects "système" pour rendre le code exécutable et performant, en particulier la gestion mémoire et l'optimisation des gradients.

### 5.1 Gradient Checkpointing (Optimisation Mémoire)

C'est une fonctionnalité indispensable pour entraîner un TRM sur GPU grand public. Puisque nous déroulons la boucle sur $T=16$ pas, nous stockons $16 \times$ les activations en mémoire pour la rétropropagation. Cela peut faire exploser la VRAM.

L'utilisation de `torch.utils.checkpoint` permet de ne pas stocker les activations intermédiaires dans le bloc récursif, mais de les recalculer à la volée lors de la passe arrière (Backprop). Cela réduit la consommation mémoire de $O(T \times L)$ à $O(T + L)$, au prix d'une légère augmentation du temps de calcul (30% environ).

**Directive :** Enveloppez l'appel `self.transformer_block(combined_input)` dans une fonction de checkpointing.

### 5.2 Optimiseur et Hyperparamètres d'Entraînement

* **Optimiseur :** AdamW est le standard de facto pour les Transformers.
* **Learning Rate :** $1e-4$ à $3e-4$.
* **Weight Decay :** $0.1$ (Les petits modèles bénéficient d'une régularisation forte).
* **Batch Size :** Maximisez-le grâce au Gradient Checkpointing. Un grand batch stabilise les gradients bruyants issus des premières étapes de récursion.

### 5.3 Export et Inférence (ONNX)

Une fois entraîné, le modèle Python n'est pas idéal pour un moteur d'échecs (souvent en C++ pour la recherche AlphaBeta). L'architecture TRM, étant un "unrolled loop", peut être exportée de deux manières :

* **Export du Noyau Seul :** On exporte juste le `TransformerBlock`. Le moteur C++ gère la boucle `for` et les buffers $y, z$. C'est l'approche la plus flexible (permet de changer le nombre d'itérations sans ré-export).
* **Export Déroulé (Trace) :** On exporte le graphe complet avec $T=10$ fixe. Plus simple à intégrer mais moins flexible.

**Recommandation :** Visez l'export du noyau seul. Cela permet d'ajuster la force du moteur (nombre de pas de réflexion) dynamiquement sans changer le fichier de poids.

## 6. Todo List Actionnable Organisée par Fonctionnalités

Voici la feuille de route opérationnelle, dérivée de l'analyse ci-dessus. Elle est structurée pour une intégration progressive des fonctionnalités.

### Fonctionnalité A : Infrastructure de Données (Feature: Data Pipeline)
**Objectif :** Transformer le fichier H5 brut en flux de tenseurs optimisés.

* **[Critique] Validateur H5 :** Créer un script d'inspection (`inspect_h5.py`) pour confirmer les dimensions et les targets. Vérifier l'absence de valeurs NaN.
* **Classe ChessDataset :** Implémenter l'héritage `torch.utils.data.Dataset`.
  * **Action :** Utiliser `h5py` avec l'option `swmr=True` (Single Writer Multiple Reader) si lecture concurrente.
  * **Optimisation :** Implémenter un cache RAM partiel si le dataset tient en mémoire (>32 Go RAM).
* **Mappage UCI :** Intégrer la conversion Move $\leftrightarrow$ Index (0-1967).
  * **Action :** Utiliser la logique `python-chess` pour générer ce mapping une fois et le geler.
* **Dataloader Optimisé :** Configurer `pin_memory=True` et `num_workers=4` pour nourrir le GPU sans latence.

### Fonctionnalité B : Noyau du Modèle TRM (Feature: Model Core)
**Objectif :** Implémenter la classe ChessTRM avec support récursif.

* **Encodage Positionnel :** Implémenter `LearnablePositionalEmbedding` (taille $64 \times d_{model}$).
* **Fusionneur de Flux (StreamCombiner) :** Coder la logique d'addition : $x_{emb} + \text{Proj}(y) + z$.
  * **Action :** Assurer que le broadcasting de $y$ (vecteur global vers grille spatiale) est correct.
* **Bloc Transformer :** Implémenter le bloc standard avec Pre-Norm.
  * **Attention :** Vérifier que `batch_first=True` dans `MultiheadAttention`.
* **Méthode forward Déroulée :** Coder la boucle temporelle qui accumule les sorties intermédiaires [`logits_t1`, `logits_t2`,...].
* **Gestionnaire d'État (Stateful Inference) :** Ajouter une méthode `reset_state()` pour vider $z$ entre deux parties lors de l'inférence.

### Fonctionnalité C : Entraînement et Supervision (Feature: Training Loop)
**Objectif :** Implémenter la DIS et la boucle d'optimisation.

* **Loss DIS Pondérée :** Créer une classe `DISLoss(nn.Module)`.
  * **Action :** Implémenter la formule $\sum w_t \mathcal{L}_{CE}$.
  * **Paramètre :** Ajouter un argument pour le type de schedule (linéaire/uniforme).
* **Intégration Gradient Checkpointing :** Activer `use_reentrant=False` (nouveau défaut PyTorch) pour économiser la VRAM.
* **Métriques par Étape :** Ne pas logger juste la loss finale. Logger Accuracy @ Step 1, Accuracy @ Step 5, Accuracy @ Step 16.
  * **Insight :** Cela permet de vérifier si le modèle apprend réellement à améliorer sa réponse ou s'il stagne dès le début.

### Fonctionnalité D : Utilitaires et Inférence (Feature: Inference Tools)
**Objectif :** Rendre le modèle jouable.

* **Adaptateur FEN :** Une fonction `fen_to_tensor(fen_str)` pour tester le modèle manuellement sur des positions spécifiques.
* **Selecteur de Coup (Greedy/Sampling) :** Implémenter la logique de choix de coup basée sur les logits finaux.
* **Export ONNX :** Script de conversion du noyau `TransformerBlock` pour utilisation future dans un moteur compilé.

## 7. Conclusion

L'implémentation de la classe ChessTRM représente un défi d'ingénierie fascinant, basculant la complexité de l'architecture spatiale vers la dynamique temporelle. En suivant cette spécification — notamment l'adoption rigoureuse des 19 plans d'entrée, la fusion additive des flux et, surtout, l'application de la Supervision d'Amélioration Profonde avec pondération linéaire — vous construirez un moteur capable d'exhiber des comportements tactiques profonds avec une empreinte mémoire minime.

Les données sont prêtes (H5). L'architecture est définie. La priorité immédiate est le codage du Noyau Récursif (Feature B) et la validation de la circulation des gradients à travers la boucle temporelle via la Loss DIS (Feature C). Une fois ces deux piliers stabilisés, le modèle commencera non pas simplement à apprendre des coups, mais à apprendre à réfléchir.
