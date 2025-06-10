document.addEventListener('DOMContentLoaded', () => {
    // Liste complète des 40 concepts, classés par catégories
    const concepts = [
        // Catégorie: Fondamentaux du Machine Learning
        {
            id: 'ml',
            category: 'Fondamentaux du Machine Learning',
            title: '1. Machine Learning',
            shortDesc: 'Algorithmes fondamentaux, statistiques et techniques d\'entraînement de modèles.',
            details: {
                useCases: [
                    'Détection de spam (ex: classificateur Naive Bayes)',
                    'Prédiction de l\'attrition client (ex: régression logistique)',
                    'Reconnaissance d\'images (ex: Machines à Vecteurs de Support)'
                ],
                projectExamples: 'Système de recommandation de Netflix, classement de recherche de Google.',
                rex: 'La qualité des données est primordiale ; "garbage in, garbage out". Le surapprentissage est un défi courant nécessitant une validation rigoureuse.',
                goodPractices: [
                    'Prétraitement approfondi des données.',
                    'Validation croisée pour une évaluation robuste du modèle.'
                ],
                metrics: [
                    'Précision, Rappel, F1-Score.',
                    'Erreur Quadratique Moyenne (MSE), R-carré.'
                ]
            }
        },
        {
            id: 'sl',
            category: 'Fondamentaux du Machine Learning',
            title: '2. Apprentissage Supervisé',
            shortDesc: 'Apprend des modèles à partir de données étiquetées pour faire des prédictions ou des classifications.',
            details: {
                useCases: [
                    'Classification d\'images avec des étiquettes (chat/chien)',
                    'Régression pour prédire les prix des maisons',
                    'Détection de spam (avec des emails étiquetés)'
                ],
                projectExamples: 'Modèles de prédiction météorologique, systèmes de reconnaissance vocale utilisant des paires audio-texte.',
                rex: 'Nécessite des données d\'entraînement étiquetées de haute qualité, ce qui peut être coûteux et chronophage à obtenir.',
                goodPractices: [
                    'Assurer la qualité et la quantité des données étiquetées.',
                    'Choisir le bon algorithme en fonction du type de problème (classification ou régression).'
                ],
                metrics: [
                    'Précision, Rappel, F1-Score (pour la classification).',
                    'MSE, MAE, R-squared (pour la régression).'
                ]
            }
        },
        {
            id: 'unsupervised-learning',
            category: 'Fondamentaux du Machine Learning',
            title: '3. Apprentissage Non Supervisé',
            shortDesc: 'Apprentissage de modèles à partir de données non étiquetées, en découvrant des structures cachées ou des motifs.',
            details: {
                useCases: [
                    'Clustering de clients pour la segmentation marketing.',
                    'Réduction de la dimensionnalité (ex: PCA, t-SNE).',
                    'Détection d\'anomalies dans les données de capteurs.'
                ],
                projectExamples: 'Segmentation de clients en fonction de leur comportement d\'achat, compression de données pour l\'efficacité du stockage.',
                rex: 'L\'évaluation des résultats peut être subjective car il n\'y a pas de "vérité terrain". Sensible au bruit dans les données.',
                goodPractices: [
                    'Explorer différentes techniques de clustering et de réduction de dimensionnalité.',
                    'Visualiser les résultats pour interpréter les clusters ou les projections.'
                ],
                metrics: [
                    'Silhouette Score, Davies-Bouldin Index (pour le clustering).',
                    'Erreur de reconstruction (pour les auto-encodeurs).'
                ]
            }
        },
        {
            id: 'semi-supervised-learning',
            category: 'Fondamentaux du Machine Learning',
            title: '4. Apprentissage Semi-Supervisé',
            shortDesc: 'Combine des petites quantités de données étiquetées avec de grandes quantités de données non étiquetées pour l\'entraînement.',
            details: {
                useCases: [
                    'Classification de texte avec un petit ensemble de documents étiquetés et de nombreux documents non étiquetés.',
                    'Reconnaissance d\'images lorsque l\'étiquetage est coûteux.',
                    'Amélioration de la performance d\'un modèle avec moins de données labellisées.'
                ],
                projectExamples: 'Amélioration de la classification de documents dans les entreprises avec des jeux de données partiellement étiquetés.',
                rex: 'Le défi est de bien utiliser les données non étiquetées pour ne pas introduire de bruit. Peut être plus complexe à mettre en œuvre que le supervisé pur.',
                goodPractices: [
                    'Utiliser des techniques comme l\'auto-entraînement ou la co-formation.',
                    'S\'assurer que les données non étiquetées sont pertinentes pour le problème.'
                ],
                metrics: [
                    'Amélioration de la précision par rapport à l\'apprentissage purement supervisé avec les mêmes données étiquetées.',
                    'Réduction du coût d\'étiquetage.'
                ]
            }
        },
        {
            id: 'rl',
            category: 'Fondamentaux du Machine Learning',
            title: '5. Apprentissage par Renforcement',
            shortDesc: 'Les agents apprennent à prendre des décisions optimales par essai-erreur dans un environnement.',
            details: {
                useCases: [
                    'Jeux (ex: AlphaGo pour le Go, AlphaStar pour StarCraft II)',
                    'Robotique (apprentissage de la marche, de la préhension)',
                    'Optimisation de la gestion du trafic'
                ],
                projectExamples: 'DeepMind\'s AlphaGo, OpenAI Five (Dota 2).',
                rex: 'Le défi de la "récompense sparse" (peu de retours sur l\'action). Peut nécessiter un grand nombre d\'interactions avec l\'environnement.',
                goodPractices: [
                    'Définition claire des fonctions de récompense.',
                    'Utilisation de simulateurs pour l\'entraînement initial.'
                ],
                metrics: [
                    'Valeur de la fonction de récompense cumulée.',
                    'Taux de succès des tâches.'
                ]
            }
        },
        {
            id: 'bl',
            category: 'Fondamentaux du Machine Learning',
            title: '6. Apprentissage Bayésien',
            shortDesc: 'Incorpore l\'incertitude en utilisant des approches de modèles probabilistes.',
            details: {
                useCases: [
                    'Filtrage anti-spam (Naive Bayes)',
                    'Diagnostic médical (mise à jour des probabilités de maladies)',
                    'Modélisation de l\'incertitude dans les prévisions'
                ],
                projectExamples: 'Moteurs de recommandation personnalisés, systèmes d\'apprentissage bayésiens pour la robotique.',
                rex: 'Peut être coûteux en calcul pour des modèles complexes ou de grandes données. L\'estimation des probabilités a priori peut être difficile.',
                goodPractices: [
                    'Bien définir les distributions de probabilité a priori.',
                    'Utiliser des méthodes d\'inférence MCMC pour les modèles complexes.'
                ],
                metrics: [
                    'Log-vraisemblance, Erreur de prédiction avec intervalles de confiance.',
                    'Calibration des probabilités.'
                ]
            }
        },
        {
            id: 'active-learning',
            category: 'Fondamentaux du Machine Learning',
            title: '7. Apprentissage Actif',
            shortDesc: 'Stratégie où le modèle sélectionne activement les données non étiquetées les plus informatives à étiqueter.',
            details: {
                useCases: [
                    'Réduction du coût d\'étiquetage manuel des données.',
                    'Accélérer le processus d\'entraînement des modèles lorsque l\'étiquetage est coûteux.',
                    'Améliorer la performance du modèle avec moins de données étiquetées.'
                ],
                projectExamples: 'Systèmes de diagnostic médical où l\'IA demande l\'avis d\'experts sur des cas ambigus, tri de documents juridiques.',
                rex: 'Le choix de la stratégie de sélection des exemples les plus informatifs est crucial. Peut nécessiter une boucle d\'interaction humaine.',
                goodPractices: [
                    'Définir une fonction de "requête" (query strategy) pour choisir les échantillons.',
                    'Intégrer les retours humains dans la boucle d\'entraînement.'
                ],
                metrics: [
                    'Nombre d\'exemples étiquetés nécessaires pour atteindre une certaine performance.',
                    'Coût total d\'étiquetage.'
                ]
            }
        },
        {
            id: 'ensemble-methods',
            category: 'Fondamentaux du Machine Learning',
            title: '8. Méthodes d\'Ensemble',
            shortDesc: 'Combinaison de plusieurs modèles d\'apprentissage pour améliorer la performance prédictive et la robustesse.',
            details: {
                useCases: [
                    'Réduction de la variance et des biais dans les prédictions.',
                    'Amélioration de la précision sur des problèmes complexes.',
                    'Ex: Random Forest, Gradient Boosting (XGBoost, LightGBM).'
                ],
                projectExamples: 'Victoires dans les compétitions Kaggle, systèmes de détection de fraude haute performance.',
                rex: 'Peut être plus coûteux en calcul que des modèles uniques. L\'interprétabilité peut être réduite.',
                goodPractices: [
                    'Utiliser des modèles de base diversifiés pour le bagging et le boosting.',
                    'Valider attentivement la combinaison des modèles pour éviter le surapprentissage.'
                ],
                metrics: [
                    'Amélioration significative des métriques de performance par rapport aux modèles individuels.'
                ]
            }
        },

        // Catégorie: Réseaux Neuronaux & Apprentissage Profond
        {
            id: 'dl',
            category: 'Réseaux Neuronaux & Apprentissage Profond',
            title: '9. Deep Learning',
            shortDesc: 'Réseaux neuronaux hiérarchiques apprenant automatiquement des représentations complexes.',
            details: {
                useCases: [
                    'Assistants vocaux (ex: Siri, Alexa)',
                    'Conduite autonome (détection d\'objets)',
                    'Analyse d\'images médicales (détection de tumeurs)'
                ],
                projectExamples: 'AlphaGo, Autopilot de Tesla.',
                rex: 'Nécessite de grands ensembles de données et des ressources de calcul importantes (GPU). Le débogage des réseaux neuronaux profonds peut être complexe.',
                goodPractices: [
                    'Utilisation de modèles pré-entraînés (transfert learning).',
                    'Ajustement minutieux des hyperparamètres.'
                ],
                metrics: [
                    'Précision, Perte (ex: Cross-entropy loss).',
                    'F1-Score, AUC-ROC.'
                ]
            }
        },
        {
            id: 'nn',
            category: 'Réseaux Neuronaux & Apprentissage Profond',
            title: '10. Réseaux Neuronaux',
            shortDesc: 'Architectures en couches modélisant efficacement les relations non linéaires avec précision.',
            details: {
                useCases: [
                    'Reconnaissance d\'écriture manuscrite (ex: ensemble de données MNIST)',
                    'Prédiction boursière',
                    'Détection de fraude'
                ],
                projectExamples: 'Premiers systèmes de classification d\'images, reconnaissance vocale dans des applications de base.',
                rex: 'Les problèmes de gradients évanescents/explosifs étaient courants avant des avancées comme ReLU. Peut être gourmand en calcul pour de grands réseaux.',
                goodPractices: [
                    'Fonctions d\'activation appropriées (ex: ReLU).',
                    'Techniques d\'initialisation des poids.'
                ],
                metrics: [
                    'Valeurs de la fonction de coût (ex: MSE, Cross-entropy).',
                    'Précision de classification.'
                ]
            }
        },
        {
            id: 'cnn',
            category: 'Réseaux Neuronaux & Apprentissage Profond',
            title: '11. Réseaux Neuronaux Convolutifs (CNNs)',
            shortDesc: 'Spécialisés dans le traitement des données de type grille comme les images, en détectant des motifs spatiaux.',
            details: {
                useCases: [
                    'Classification d\'images (identifier des objets dans des photos).',
                    'Détection et localisation d\'objets dans une image.',
                    'Segmentation d\'images (identifier chaque pixel appartenant à un objet).'
                ],
                projectExamples: 'Applications de reconnaissance faciale, systèmes de vision pour les voitures autonomes.',
                rex: 'Nécessitent de grandes quantités de données étiquetées pour l\'entraînement. Sensibles à la rotation ou à la mise à l\'échelle des objets.',
                goodPractices: [
                    'Utiliser des couches de convolution, de pooling et des fonctions d\'activation ReLU.',
                    'Employer l\'augmentation de données pour augmenter la robustesse aux variations.'
                ],
                metrics: [
                    'Précision de classification, mAP (Mean Average Precision) pour la détection d\'objets.',
                    'IoU (Intersection over Union) pour la segmentation.'
                ]
            }
        },
        {
            id: 'rnn',
            category: 'Réseaux Neuronaux & Apprentissage Profond',
            title: '12. Réseaux Neuronaux Récurrents (RNNs)',
            shortDesc: 'Conçus pour traiter des séquences de données (texte, parole, séries temporelles) en utilisant des boucles internes.',
            details: {
                useCases: [
                    'Prédiction de mots suivants dans une phrase.',
                    'Traduction automatique (avant les Transformers).',
                    'Reconnaissance de la parole.'
                ],
                projectExamples: 'Anciens modèles de traduction automatique, systèmes de dictée vocale.',
                rex: 'Souffrent du problème des gradients évanescents/explosifs sur les longues séquences. Plus lents à entraîner que les Transformers.',
                goodPractices: [
                    'Utiliser des variantes comme les LSTMs (Long Short-Term Memory) ou les GRUs (Gated Recurrent Unit) pour atténuer les problèmes de gradient.',
                    'Tronquer les séquences pour des raisons de performance.'
                ],
                metrics: [
                    'Perplexité (pour les modèles de langage), BLEU score (pour la traduction).',
                    'Précision de reconnaissance (pour la parole).'
                ]
            }
        },
        {
            id: 'transformers',
            category: 'Réseaux Neuronaux & Apprentissage Profond',
            title: '13. Transformers',
            shortDesc: 'Architecture basée sur l\'auto-attention propulsant les modèles d\'IA modernes.',
            details: {
                useCases: [
                    'Traduction automatique neuronale',
                    'Modélisation du langage',
                    'Compréhension de la parole'
                ],
                projectExamples: 'BERT, GPT (basés sur l\'architecture Transformer).',
                rex: 'Très gourmands en ressources de calcul pour l\'entraînement de grands modèles. La compréhension de l\'attention peut être complexe.',
                goodPractices: [
                    'Tirer parti des modèles Transformer pré-entraînés.',
                    'Utiliser des techniques d\'optimisation comme le quantizing pour le déploiement.'
                ],
                metrics: [
                    'Précision, Rappel, F1-Score pour les tâches NLP.',
                    'Vitesse d\'inférence.'
                ]
            }
        },
        {
            id: 'gm',
            category: 'Réseaux Neuronaux & Apprentissage Profond',
            title: '14. Modèles Génératifs',
            shortDesc: 'Création de nouveaux échantillons de données à partir de données apprises.',
            details: {
                useCases: [
                    'Génération d\'images (visages, paysages)',
                    'Création de musique ou de texte',
                    'Augmentation de données pour l\'entraînement de modèles'
                ],
                projectExamples: 'StyleGAN (génération de visages réalistes), DALL-E (texte-vers-image).',
                rex: 'Le mode collapse (génération de données peu variées) est un problème courant avec les GANs. L\'évaluation de la qualité des données générées est subjective.',
                goodPractices: [
                    'Utilisation de Generative Adversarial Networks (GANs) ou de VAEs (Variational Autoencoders).',
                    'Équilibrer l\'entraînement du générateur et du discriminateur dans les GANs.'
                ],
                metrics: [
                    'Inception Score (IS), Fréchet Inception Distance (FID) pour les images.',
                    'Cohérence et pertinence pour le texte généré.'
                ]
            }
        },
        {
            id: 'gan',
            category: 'Réseaux Neuronaux & Apprentissage Profond',
            title: '15. Generative Adversarial Networks (GANs)',
            shortDesc: 'Deux réseaux neuronaux (générateur et discriminateur) s\'affrontent pour générer des données réalistes.',
            details: {
                useCases: [
                    'Génération de visages humains photoréalistes (ex: ThisPersonDoesNotExist.com).',
                    'Création d\'œuvres d\'art numériques ou de styles artistiques.',
                    'Augmentation de données pour l\'entraînement de modèles (création de données synthétiques).'
                ],
                projectExamples: 'StyleGAN, GauGAN (NVIDIA pour la synthèse d\'images à partir de croquis).',
                rex: 'Très difficile à entraîner en raison de la "course" entre le générateur et le discriminateur, souvent instable (mode collapse).',
                goodPractices: [
                    'Utiliser des techniques d\'entraînement avancées (Wasserstein GANs, Progressive GANs).',
                    'Surveiller les métriques de diversité et de qualité des données générées.'
                ],
                metrics: [
                    'Fréchet Inception Distance (FID), Inception Score (IS).',
                    'Évaluation humaine de la qualité visuelle ou sémantique des données générées.'
                ]
            }
        },
        {
            id: 'transfer-learning',
            category: 'Réseaux Neuronaux & Apprentissage Profond',
            title: '16. Apprentissage par Transfert',
            shortDesc: 'Réutilisation d\'un modèle pré-entraîné sur une nouvelle tâche connexe, économisant temps et ressources.',
            details: {
                useCases: [
                    'Utiliser un modèle de reconnaissance d\'images entraîné sur ImageNet pour une tâche de classification d\'images médicales spécifiques.',
                    'Adapter un grand modèle de langage pour des tâches de résumé de texte spécifiques à un domaine.',
                ],
                projectExamples: 'Utilisation de VGG16 ou ResNet pour la classification de nouvelles catégories d\'images, adaptation de BERT pour des tâches de classification de texte spécialisées.',
                rex: 'Le choix du modèle pré-entraîné pertinent est crucial. Un déséquilibre entre les domaines source et cible peut limiter les gains.',
                goodPractices: [
                    'Dégeler les couches finales du modèle pré-entraîné et les entraîner sur les nouvelles données.',
                    'Utiliser un taux d\'apprentissage plus faible pour les couches gelées et un taux plus élevé pour les nouvelles couches.'
                ],
                metrics: [
                    'Amélioration de la performance sur la nouvelle tâche par rapport à un entraînement "from scratch".',
                    'Réduction du temps d\'entraînement et de la quantité de données requises.'
                ]
            }
        },
        {
            id: 'fine-tuning',
            category: 'Réseaux Neuronaux & Apprentissage Profond',
            title: '17. Fine-Tuning de Modèles',
            shortDesc: 'Personnalise les modèles pré-entraînés pour des tâches spécifiques à un domaine.',
            details: {
                useCases: [
                    'Adapter un LLM généraliste à un domaine juridique ou médical',
                    'Personnaliser un modèle de vision par ordinateur pour détecter des types spécifiques de défauts industriels',
                    'Adapter un modèle de reconnaissance vocale à un accent particulier'
                ],
                projectExamples: 'Affinement de BERT pour la classification de documents financiers, entraînement d\'un GPT sur des données de service client.',
                rex: 'Nécessite un ensemble de données d\'entraînement plus petit mais de haute qualité pour la tâche spécifique. Le surapprentissage peut toujours être un problème.',
                goodPractices: [
                    'Utiliser des taux d\'apprentissage plus petits que pour l\'entraînement initial.',
                    'Geler certaines couches du modèle pré-entraîné pour éviter le surapprentissage.'
                ],
                metrics: [
                    'Amélioration de la performance sur la tâche spécifique (précision, F1-Score).',
                    'Réduction du temps et des ressources nécessaires par rapport à un entraînement à partir de zéro.'
                ]
            }
        },
        {
            id: 'multimodal',
            category: 'Réseaux Neuronaux & Apprentissage Profond',
            title: '18. Modèles Multimodaux',
            shortDesc: 'Traitent et génèrent des données à travers plusieurs types (images, vidéos, texte).',
            details: {
                useCases: [
                    'Génération de légendes d\'images (décrire une image avec du texte)',
                    'Recherche d\'images basée sur une description textuelle',
                    'Génération de vidéos à partir de texte'
                ],
                projectExamples: 'CLIP (OpenAI), DALL-E 2, Flamingo (DeepMind).',
                rex: 'La fusion et la cohérence des informations provenant de différentes modalités sont complexes. Nécessite des ensembles de données multimodales massifs.',
                goodPractices: [
                    'Concevoir des architectures capables de comprendre les relations inter-modalités.',
                    'Aligner les représentations des différentes modalités dans un espace commun.'
                ],
                metrics: [
                    'Qualité et pertinence des sorties générées sur plusieurs modalités.',
                    'Évaluation humaine pour la cohérence multimodale.'
                ]
            }
        },
        {
            id: 'embeddings',
            category: 'Réseaux Neuronaux & Apprentissage Profond',
            title: '19. Embeddings',
            shortDesc: 'Transforme l\'entrée en formats vectoriels lisibles par la machine.',
            details: {
                useCases: [
                    'Représentation de mots pour le TLN (Word2Vec, GloVe)',
                    'Représentation d\'images pour la recherche de similarité',
                    'Représentation de graphes ou de nœuds de réseau'
                ],
                projectExamples: 'Utilisation d\'embeddings de mots pour améliorer la recherche sémantique, représentation d\'articles pour la recommandation.',
                rex: 'Le choix de la taille de l\'embedding et de la méthode d\'apprentissage peut affecter la performance. L\'interprétabilité des embeddings est limitée.',
                goodPractices: [
                    'Utiliser des embeddings pré-entraînés lorsque cela est possible.',
                    'Visualiser les embeddings (ex: avec t-SNE) pour comprendre leurs relations.'
                ],
                metrics: [
                    'Performance du modèle en aval utilisant les embeddings.',
                    'Similarité cosinus entre vecteurs pour la mesure de similarité sémantique.'
                ]
            }
        },
        {
            id: 'vector-search',
            category: 'Réseaux Neuronaux & Apprentissage Profond',
            title: '20. Recherche Vectorielle',
            shortDesc: 'Trouve des éléments similaires en utilisant des embeddings vectoriels denses.',
            details: {
                useCases: [
                    'Recherche sémantique (trouver des documents ou images similaires à une requête)',
                    'Systèmes de recommandation basés sur la similarité',
                    'Détection d\'anomalies (trouver des points de données éloignés des clusters)'
                ],
                projectExamples: 'Moteurs de recherche basés sur la pertinence sémantique, bases de données vectorielles (Pinecone, Milvus).',
                rex: 'La scalabilité de la recherche sur des milliards de vecteurs est un défi. La qualité des embeddings impacte directement la pertinence des résultats.',
                goodPractices: [
                    'Utiliser des indexes Approximate Nearest Neighbor (ANN) pour la scalabilité.',
                    'Mettre à jour régulièrement les embeddings à mesure que les données évoluent.'
                ],
                metrics: [
                    'Recall@k (combien de voisins pertinents sont trouvés parmi les k premiers résultats).',
                    'Latence de recherche.'
                ]
            }
        },

        // Catégorie: Applications & Domaines Spécifiques
        {
            id: 'nlp',
            category: 'Applications & Domaines Spécifiques',
            title: '21. Traitement du Langage Naturel (TLN)',
            shortDesc: 'Techniques pour traiter et comprendre le texte en langage naturel.',
            details: {
                useCases: [
                    'Traduction automatique (ex: Google Translate)',
                    'Analyse de sentiments dans les avis clients',
                    'Chatbots et assistants virtuels'
                ],
                projectExamples: 'Google Assistant, ChatGPT, Grammarly.',
                rex: 'La gestion de l\'ambiguïté linguistique et des nuances est difficile. Nécessite souvent de grandes quantités de données textuelles labellisées.',
                goodPractices: [
                    'Utilisation de modèles de langage pré-entraînés (ex: BERT, GPT).',
                    'Techniques de normalisation du texte (stemming, lemmatisation).'
                ],
                metrics: [
                    'Précision, Rappel, F1-Score pour la classification de texte.',
                    'BLEU score pour la traduction automatique.'
                ]
            }
        },
        {
            id: 'cv',
            category: 'Applications & Domaines Spécifiques',
            title: '22. Vision par Ordinateur',
            shortDesc: 'Algorithmes interprétant et analysant efficacement les données visuelles.',
            details: {
                useCases: [
                    'Reconnaissance faciale (déverrouillage de téléphone)',
                    'Détection d\'objets dans les véhicules autonomes',
                    'Inspection de qualité industrielle'
                ],
                projectExamples: 'Systèmes de surveillance par caméra intelligente, reconnaissance de plaques d\'immatriculation.',
                rex: 'Sensibilité aux conditions d\'éclairage et aux occlusions. La nécessité de vastes ensembles de données annotées pour l\'entraînement.',
                goodPractices: [
                    'Utilisation de Réseaux Neuronaux Convolutifs (CNN).',
                    'Augmentation de données pour améliorer la robustesse du modèle.'
                ],
                metrics: [
                    'Précision (Accuracy), Intersection over Union (IoU) pour la détection d\'objets.',
                    'Mean Average Precision (mAP).'
                ]
            }
        },
        {
            id: 'llm',
            category: 'Applications & Domaines Spécifiques',
            title: '23. Grands Modèles de Langage (LLM)',
            shortDesc: 'Génère du texte similaire à celui humain à l\'aide de données pré-entraînées massives.',
            details: {
                useCases: [
                    'Rédaction de contenu (articles, e-mails)',
                    'Résumé de texte',
                    'Réponse aux questions complexes'
                ],
                projectExamples: 'ChatGPT, Bard, GPT-4.',
                rex: 'Risque d\'hallucinations (génération d\'informations fausses mais plausibles). Coût de calcul élevé pour l\'entraînement et le déploiement.',
                goodPractices: [
                    'Utilisation de techniques de prompt engineering efficaces.',
                    'Fine-tuning sur des données spécifiques au domaine.'
                ],
                metrics: [
                    'Perplexité, BLEU, ROUGE pour l\'évaluation du texte.',
                    'Évaluation humaine pour la cohérence et la pertinence.'
                ]
            }
        },
        {
            id: 'pe',
            category: 'Applications & Domaines Spécifiques',
            title: '24. Ingénierie des Prompts (Prompt Engineering)',
            shortDesc: 'Création d\'entrées efficaces pour guider les sorties des modèles génératifs.',
            details: {
                useCases: [
                    'Obtention de réponses précises et pertinentes des LLM',
                    'Génération d\'images spécifiques avec des modèles texte-image',
                    'Affinement du style et du ton du texte généré'
                ],
                projectExamples: 'Optimisation des requêtes pour ChatGPT afin d\'obtenir des résumés précis ou des codes fonctionnels.',
                rex: 'Peut être un processus itératif et nécessiter de l\'intuition. La subtilité des mots peut avoir un impact énorme sur la sortie.',
                goodPractices: [
                    'Être précis et concis dans les instructions.',
                    'Utiliser des exemples pour guider le modèle.',
                    'Spécifier le format de sortie souhaité.'
                ],
                metrics: [
                    'Pertinence et cohérence des sorties générées.',
                    'Réduction du nombre d\'itérations pour obtenir le résultat souhaité.'
                ]
            }
        },
        {
            id: 'ai-agents',
            category: 'Applications & Domaines Spécifiques',
            title: '25. Agents IA',
            shortDesc: 'Systèmes autonomes qui perçoivent, décident et agissent.',
            details: {
                useCases: [
                    'Bots de trading financier',
                    'Assistants virtuels (ex: Google Duplex)',
                    'Robots de service autonomes'
                ],
                projectExamples: 'Robots nettoyeurs de sol autonomes, agents d\'IA dans les jeux vidéo pour des PNJ réalistes.',
                rex: 'La gestion des interactions avec le monde réel peut être complexe. Assurer la sécurité et la fiabilité de l\'agent dans des environnements dynamiques.',
                goodPractices: [
                    'Définir clairement les objectifs et les contraintes de l\'agent.',
                    'Tester rigoureusement l\'agent dans divers scénarios.'
                ],
                metrics: [
                    'Taux de succès des tâches, temps d\'achèvement des tâches.',
                    'Efficacité énergétique (pour les agents physiques).'
                ]
            }
        },
        {
            id: 'robotics-ai',
            category: 'Applications & Domaines Spécifiques',
            title: '26. IA & Robotique',
            shortDesc: 'Application de l\'intelligence artificielle pour permettre aux robots de percevoir, raisonner, apprendre et interagir avec leur environnement.',
            details: {
                useCases: [
                    'Robots collaboratifs dans l\'industrie.',
                    'Robots de service (nettoyage, livraison).',
                    'Drones autonomes pour l\'inspection ou la surveillance.'
                ],
                projectExamples: 'Boston Dynamics (robots humanoïdes et quadrupèdes), robots chirurgicaux Da Vinci.',
                rex: 'Les défis liés à la manipulation d\'objets, à la navigation dans des environnements complexes et à l\'interaction homme-robot en toute sécurité.',
                goodPractices: [
                    'Intégrer la vision par ordinateur pour la perception de l\'environnement.',
                    'Utiliser l\'apprentissage par renforcement pour l\'apprentissage de tâches complexes.',
                    'Prioriser la sécurité dans la conception et le déploiement des robots.'
                ],
                metrics: [
                    'Taux de succès des tâches robotiques, précision de la navigation.',
                    'Temps d\'exécution des tâches, robustesse à des conditions variables.'
                ]
            }
        },
        {
            id: 'digital-twins',
            category: 'Applications & Domaines Spécifiques',
            title: '27. Jumeaux Numériques',
            shortDesc: 'Représentation virtuelle d\'un objet, système ou processus physique, mise à jour en temps réel avec des données du monde réel, souvent augmentée par l\'IA.',
            details: {
                useCases: [
                    'Simulation du comportement d\'une usine avant sa construction.',
                    'Maintenance prédictive d\'équipements complexes (turbines, moteurs).',
                    'Optimisation des performances d\'une ville intelligente.'
                ],
                projectExamples: 'Siemens MindSphere pour les usines numériques, jumeaux numériques pour l\'ingénierie aéronautique.',
                rex: 'Nécessite des données en temps réel précises. La complexité de la modélisation fidèle du monde physique et de l\'intégration des systèmes d\'IA.',
                goodPractices: [
                    'S\'assurer de la qualité et de la fréquence de la collecte de données.',
                    'Utiliser des modèles d\'IA pour l\'analyse prédictive et la simulation.'
                ],
                metrics: [
                    'Précision des prédictions de défaillance, réduction des temps d\'arrêt.',
                    'Efficacité opérationnelle améliorée.'
                ]
            }
        },
        {
            id: 'edge-ai',
            category: 'Applications & Domaines Spécifiques',
            title: '28. IA Embarquée (Edge AI)',
            shortDesc: 'Déploiement de l\'intelligence artificielle directement sur les périphériques (caméras, capteurs, smartphones) plutôt que dans le cloud.',
            details: {
                useCases: [
                    'Traitement d\'images en temps réel sur une caméra de sécurité.',
                    'Assistants vocaux fonctionnant hors ligne sur smartphone.',
                    'Maintenance prédictive sur des machines industrielles sans connectivité constante.'
                ],
                projectExamples: 'Google Coral (TPU pour l\'Edge AI), applications de traitement du langage naturel sur smartphones.',
                rex: 'Limitations des ressources matérielles (puissance de calcul, mémoire, batterie). Nécessite des modèles optimisés et légers.',
                goodPractices: [
                    'Utiliser des techniques de compression de modèle (quantification, élagage).',
                    'Optimiser le code pour l\'inférence sur les architectures embarquées.'
                ],
                metrics: [
                    'Latence d\'inférence, consommation d\'énergie.',
                    'Taille du modèle, performance du modèle sur le périphérique.'
                ]
            }
        },

        // Catégorie: Préparation & Optimisation des Données/Modèles
        {
            id: 'fe',
            category: 'Préparation & Optimisation des Données/Modèles',
            title: '29. Ingénierie de Caractéristiques (Feature Engineering)',
            shortDesc: 'Conception de caractéristiques informatives pour améliorer significativement la performance du modèle.',
            details: {
                useCases: [
                    'Création de nouvelles variables à partir de données brutes',
                    'Combinaison de caractéristiques existantes',
                    'Sélection des caractéristiques les plus pertinentes'
                ],
                projectExamples: 'Amélioration de la détection de fraude en créant des caractéristiques comme le nombre de transactions par heure.',
                rex: 'Processus chronophage et nécessitant une expertise métier. Un mauvais feature engineering peut dégrader les performances du modèle.',
                goodPractices: [
                    'Comprendre le domaine d\'application pour créer des caractéristiques pertinentes.',
                    'Utiliser des techniques de sélection de caractéristiques pour réduire la dimensionnalité.'
                ],
                metrics: [
                    'Amélioration de la précision/performance du modèle.',
                    'Réduction du temps d\'entraînement.'
                ]
            }
        },
        {
            id: 'data-augmentation',
            category: 'Préparation & Optimisation des Données/Modèles',
            title: '30. Augmentation de Données',
            shortDesc: 'Technique pour augmenter artificiellement la taille d\'un jeu de données d\'entraînement en créant des versions modifiées des données existantes.',
            details: {
                useCases: [
                    'Augmenter les images en les faisant pivoter, en les recadrant ou en ajustant la luminosité.',
                    'Créer des variantes de phrases pour l\'entraînement de modèles NLP.',
                    'Réduire le surapprentissage et améliorer la généralisation du modèle.'
                ],
                projectExamples: 'Amélioration de la robustesse des modèles de classification d\'images face à des variations de position ou d\'éclairage.',
                rex: 'Les augmentations doivent être réalistes et pertinentes pour le problème. Une augmentation excessive peut introduire du bruit.',
                goodPractices: [
                    'Appliquer des transformations qui simulent des variations réalistes dans les données du monde réel.',
                    'Utiliser des bibliothèques dédiées (ex: Keras ImageDataGenerator, Albumentations).'
                ],
                metrics: [
                    'Amélioration de la précision sur l\'ensemble de test.',
                    'Réduction du surapprentissage.'
                ]
            }
        },
        {
            id: 'hyperparameter-tuning',
            category: 'Préparation & Optimisation des Données/Modèles',
            title: '31. Optimisation des Hyperparamètres',
            shortDesc: 'Processus de recherche des meilleurs hyperparamètres pour un modèle d\'IA afin d\'optimiser ses performances.',
            details: {
                useCases: [
                    'Déterminer le nombre optimal de couches dans un réseau neuronal.',
                    'Trouver le meilleur taux d\'apprentissage pour un algorithme d\'optimisation.',
                    'Optimiser les paramètres d\'un algorithme de clustering.'
                ],
                projectExamples: 'Utilisation de Grid Search, Random Search ou Optimisation Bayésienne pour affiner les modèles de ML.',
                rex: 'Peut être très coûteux en calcul, surtout pour les grands modèles. La recherche exhaustive est souvent impraticable.',
                goodPractices: [
                    'Commencer par une recherche grossière (Random Search) puis affiner avec une recherche plus ciblée (Grid Search, Optimisation Bayésienne).',
                    'Utiliser des outils comme Optuna ou Hyperopt pour l\'automatisation.'
                ],
                metrics: [
                    'Meilleur score de validation (ex: précision, F1-Score) sur l\'ensemble de validation.',
                    'Convergence plus rapide de l\'entraînement.'
                ]
            }
        },
        {
            id: 'model-eval',
            category: 'Préparation & Optimisation des Données/Modèles',
            title: '32. Évaluation de Modèles',
            shortDesc: 'Évaluer la performance prédictive à l\'aide de techniques de validation.',
            details: {
                useCases: [
                    'Mesurer la précision d\'un modèle de classification',
                    'Évaluer l\'erreur d\'un modèle de régression',
                    'Comparer les performances de différents modèles'
                ],
                projectExamples: 'Utilisation d\'ensembles de validation et de test pour s\'assurer qu\'un modèle généralise bien à de nouvelles données.',
                rex: 'Éviter le surapprentissage en n\'évaluant pas sur les données d\'entraînement. Choisir les métriques appropriées en fonction du problème métier.',
                goodPractices: [
                    'Utiliser des ensembles de test distincts et non vus par le modèle.',
                    'Appliquer la validation croisée pour des estimations de performance plus robustes.'
                ],
                metrics: [
                    'Précision, Rappel, F1-Score, AUC-ROC (classification).',
                    'MSE, MAE, R-squared (régression).',
                    'Temps d\'inférence, utilisation de la mémoire.'
                ]
            }
        },
        {
            id: 'bias-detection',
            category: 'Préparation & Optimisation des Données/Modèles',
            title: '33. Détection de Biais',
            shortDesc: 'Identification et atténuation des biais indésirables dans les données et les modèles d\'IA.',
            details: {
                useCases: [
                    'Analyser si un modèle de recrutement favorise certains groupes démographiques.',
                    'Détecter des inégalités dans les diagnostics médicaux basés sur l\'IA.',
                    'Assurer l\'équité des systèmes de notation de crédit.'
                ],
                projectExamples: 'Outils comme AI Fairness 360 (IBM) pour l\'analyse et la mitigation des biais.',
                rex: 'Le biais peut être subtil et difficile à détecter. Nécessite une compréhension approfondie du domaine et des données.',
                goodPractices: [
                    'Collecter des données diversifiées et représentatives.',
                    'Utiliser des métriques d\'équité spécifiques en plus des métriques de performance standard.'
                ],
                metrics: [
                    'Parité démographique, égalité des chances.',
                    'Mesures de disparité des performances entre groupes.'
                ]
            }
        },
        {
            id: 'data-privacy',
            category: 'Préparation & Optimisation des Données/Modèles',
            title: '34. Confidentialité des Données',
            shortDesc: 'Techniques et principes pour protéger les informations personnelles et sensibles dans les systèmes d\'IA.',
            details: {
                useCases: [
                    'Anonymisation et pseudonymisation des données.',
                    'Confidentialité différentielle pour protéger la vie privée dans les ensembles de données agrégés.',
                    'Traitement des données sensibles pour des modèles de santé ou financiers.'
                ],
                projectExamples: 'Recherche sur les garanties de confidentialité dans l\'apprentissage fédéré, cadres de protection des données (GDPR, CCPA).',
                rex: 'Un équilibre délicat entre la protection de la vie privée et l\'utilité des données. Les réglementations sont en constante évolution.',
                goodPractices: [
                    'Mettre en œuvre des mesures de sécurité dès la conception (Privacy by Design).',
                    'Utiliser des techniques de protection des données (homomorphic encryption, secure multi-party computation).'
                ],
                metrics: [
                    'Quantification de la perte de confidentialité (ex: epsilon pour la confidentialité différentielle).',
                    'Conformité réglementaire.'
                ]
            }
        },

        // Catégorie: Déploiement & Opérations IA
        {
            id: 'ai-infra',
            category: 'Déploiement & Opérations IA',
            title: '35. Infrastructure IA',
            shortDesc: 'Déploiement de systèmes évolutifs pour prendre en charge les opérations IA.',
            details: {
                useCases: [
                    'Plateformes de calcul distribué pour l\'entraînement de modèles (ex: Kubernetes, Ray)',
                    'Systèmes de gestion de données pour le stockage et l\'accès aux données d\'IA',
                    'Pipelines MLOps pour le déploiement et la surveillance continus'
                ],
                projectExamples: 'AWS SageMaker, Google Cloud AI Platform, Azure Machine Learning.',
                rex: 'La complexité de la gestion des GPU et des ressources distribuées. Assurer la sécurité et la conformité des données à grande échelle.',
                goodPractices: [
                    'Adopter des pratiques MLOps pour automatiser le cycle de vie du ML.',
                    'Utiliser des conteneurs (Docker) et l\'orchestration (Kubernetes) pour la portabilité et la scalabilité.'
                ],
                metrics: [
                    'Coût de l\'infrastructure, temps de déploiement.',
                    'Fiabilité et disponibilité des services IA.'
                ]
            }
        },
        {
            id: 'mlo-ps',
            category: 'Déploiement & Opérations IA',
            title: '36. MLOps (Machine Learning Operations)',
            shortDesc: 'Pratiques pour déployer et maintenir des modèles de machine learning en production de manière fiable et efficace.',
            details: {
                useCases: [
                    'Automatisation du déploiement de modèles.',
                    'Surveillance continue des performances des modèles en production.',
                    'Gestion des versions de modèles et des jeux de données.'
                ],
                projectExamples: 'Mise en place de pipelines CI/CD pour le ML, plateformes MLOps (MLflow, Kubeflow).',
                rex: 'La complexité de la gestion du cycle de vie des modèles, de l\'entraînement à la production. Nécessite des compétences en développement logiciel et en ML.',
                goodPractices: [
                    'Adopter l\'automatisation à chaque étape du pipeline ML.',
                    'Mettre en place des alertes pour la dérive de modèle ou la dégradation des performances.'
                ],
                metrics: [
                    'Temps de déploiement de nouveaux modèles.',
                    'Fréquence de la dérive de modèle, temps de résolution des incidents.'
                ]
            }
        },
        {
            id: 'model-monitoring',
            category: 'Déploiement & Opérations IA',
            title: '37. Surveillance de Modèles (Model Monitoring)',
            shortDesc: 'Surveillance continue des performances des modèles d\'IA en production pour détecter la dérive de données, la dérive de modèle ou les pannes.',
            details: {
                useCases: [
                    'Détection de la dégradation des prédictions d\'un modèle de détection de fraude.',
                    'Alerte en cas de changement dans la distribution des données entrantes.',
                    'Identification des modèles obsolètes nécessitant un réentraînement.'
                ],
                projectExamples: 'Intégration avec des outils de monitoring (Prometheus, Grafana) pour les métriques de modèle.',
                rex: 'Le défi est de définir des seuils d\'alerte pertinents et de distinguer la dérive réelle du bruit. Nécessite des pipelines de données robustes.',
                goodPractices: [
                    'Surveiller les métriques de performance (précision, F1-Score) et les distributions des données.',
                    'Mettre en place des mécanismes de réentraînement automatique ou manuel en cas de dérive.'
                ],
                metrics: [
                    'Taux de dérive des données/modèles.',
                    'Temps moyen de détection et de résolution des problèmes.'
                ]
            }
        },
        {
            id: 'model-versioning',
            category: 'Déploiement & Opérations IA',
            title: '38. Versioning de Modèles',
            shortDesc: 'Gestion des différentes versions des modèles d\'IA, des données et du code pour assurer la traçabilité et la reproductibilité.',
            details: {
                useCases: [
                    'Revenir à une version précédente d\'un modèle en cas de problème en production.',
                    'Reproduire les résultats d\'une expérimentation passée.',
                    'Collaborer sur le développement de modèles au sein d\'une équipe.'
                ],
                projectExamples: 'Utilisation de registres de modèles (MLflow Model Registry, Hugging Face Hub) et de systèmes de gestion de versions (Git).',
                rex: 'La complexité de lier les versions de code, de données et de modèle. Le manque de standardisation peut entraîner des incohérences.',
                goodPractices: [
                    'Associer chaque modèle à une version de code et aux données d\'entraînement utilisées.',
                    'Utiliser des outils de versioning de modèles pour centraliser la gestion.'
                ],
                metrics: [
                    'Temps de reproduction des expérimentations.',
                    'Fiabilité des déploiements (réduction des erreurs dues aux versions incorrectes).'
                ]
            }
        },

        // Catégorie: IA Responsable & Avancée
        {
            id: 'ethical-ai',
            category: 'IA Responsable & Avancée',
            title: '39. IA Éthique',
            shortDesc: 'Conception, développement et déploiement de systèmes d\'IA qui respectent les valeurs humaines, la justice et la transparence.',
            details: {
                useCases: [
                    'Développement de lignes directrices pour l\'utilisation responsable de l\'IA.',
                    'Audit de systèmes d\'IA pour les biais et la discrimination.',
                    'Mise en place de mécanismes de redevabilité pour les décisions de l\'IA.'
                ],
                projectExamples: 'Principes d\'IA de Google, Framework de l\'OCDE sur l\'IA.',
                rex: 'Le défi est de traduire les principes éthiques abstraits en actions concrètes et mesurables. Nécessite une approche multidisciplinaire.',
                goodPractices: [
                    'Intégrer les considérations éthiques dès la phase de conception ("Ethical by Design").',
                    'Impliquer diverses parties prenantes dans le processus de développement et d\'évaluation.'
                ],
                metrics: [
                    'Mesures de l\'équité (ex: parité démographique, égalité des chances).',
                    'Transparence et explicabilité des décisions.'
                ]
            }
        },
        {
            id: 'explainable-ai-xai',
            category: 'IA Responsable & Avancée',
            title: '40. IA Explicable (XAI)',
            shortDesc: 'Développement de méthodes et de techniques permettant de comprendre et d\'interpréter les décisions des modèles d\'IA.',
            details: {
                useCases: [
                    'Comprendre pourquoi un modèle de prêt a refusé un crédit.',
                    'Identifier les caractéristiques les plus influentes dans un diagnostic médical.',
                    'Assurer la conformité réglementaire dans les systèmes d\'IA.'
                ],
                projectExamples: 'Outils comme SHAP et LIME pour l\'interprétabilité des modèles, systèmes d\'IA pour l\'audit automatisé.',
                rex: 'Compromis entre l\'interprétabilité et la performance du modèle. Peut être complexe à mettre en œuvre pour les modèles profonds.',
                goodPractices: [
                    'Utiliser des modèles intrinsèquement interprétables lorsque c\'est possible (arbres de décision, régression linéaire).',
                    'Employer des techniques post-hoc pour les modèles complexes.'
                ],
                metrics: [
                    'Fidélité de l\'explication à la décision du modèle.',
                    'Compréhensibilité de l\'explication par les humains.'
                ]
            }
        },
        {
            id: 'federated-learning',
            category: 'IA Responsable & Avancée',
            title: '41. Apprentissage Fédéré',
            shortDesc: 'Méthode d\'entraînement de modèles d\'IA où les données restent sur les appareils locaux (ex: smartphones, hôpitaux) et seuls les poids du modèle sont partagés.',
            details: {
                useCases: [
                    'Entraînement de modèles sur des données de smartphones sans que les données quittent l\'appareil (ex: clavier prédictif).',
                    'Collaboration entre hôpitaux pour entraîner un modèle sans partager de données de patients sensibles.',
                    'Amélioration de la confidentialité et de la sécurité des données.'
                ],
                projectExamples: 'Google\'s Gboard pour la prédiction de mots, consortiums de recherche médicale.',
                rex: 'La complexité de l\'agrégation des modèles. Les problèmes de non-IID (données non identiquement distribuées) entre les clients.',
                goodPractices: [
                    'Mettre en œuvre des mécanismes d\'agrégation robustes des poids du modèle.',
                    'Considérer la confidentialité différentielle pour une protection accrue.'
                ],
                metrics: [
                    'Performance du modèle agrégé.',
                    'Gain de confidentialité des données.'
                ]
            }
        },
        {
            id: 'reinforcement-learning-from-human-feedback-rlhf',
            category: 'IA Responsable & Avancée',
            title: '42. Apprentissage par Renforcement à partir du Retour Humain (RLHF)',
            shortDesc: 'Technique pour affiner les modèles d\'IA (notamment les LLM) en utilisant les préférences humaines comme signal de récompense.',
            details: {
                useCases: [
                    'Aligner les LLM avec les valeurs et intentions humaines (réduire les réponses toxiques ou inappropriées).',
                    'Améliorer la qualité et la pertinence des générations de texte.',
                    'Rendre les chatbots plus utiles et conversationnels.'
                ],
                projectExamples: 'L\'entraînement de ChatGPT et GPT-4 a largement utilisé RLHF pour améliorer la qualité de leurs réponses.',
                rex: 'Le processus d\'étiquetage des préférences humaines peut être subjectif et coûteux. La définition d\'une fonction de récompense efficace est complexe.',
                goodPractices: [
                    'Collecter des données de préférences humaines diversifiées et de haute qualité.',
                    'Utiliser des modèles de récompense bien conçus pour capturer les nuances des préférences.'
                ],
                metrics: [
                    'Évaluation humaine de la qualité, de la sécurité et de l\'utilité des réponses.',
                    'Réduction des "hallucinations" ou des biais indésirables.'
                ]
            }
        },
        {
            id: 'causality',
            category: 'IA Responsable & Avancée',
            title: '43. Causalité en IA',
            shortDesc: 'Comprendre les relations de cause à effet entre les variables, au-delà de la simple corrélation, pour une prise de décision plus robuste.',
            details: {
                useCases: [
                    'Déterminer l\'impact réel d\'une campagne marketing sur les ventes (et non juste une corrélation).',
                    'Identifier les causes profondes des défaillances de machines.',
                    'Développer des traitements médicaux basés sur des relations causales prouvées.'
                ],
                projectExamples: 'Utilisation de graphes causaux pour modéliser des systèmes complexes, cadres d\'inférence causal (ex: DoWhy).',
                rex: 'La causalité est intrinsèquement plus difficile à établir que la corrélation. Nécessite souvent des expérimentations contrôlées (tests A/B).',
                goodPractices: [
                    'Distinguer corrélation et causalité.',
                    'Utiliser des techniques d\'inférence causal pour estimer les effets de traitement.'
                ],
                metrics: [
                    'Mesure de l\'effet de traitement causal.',
                    'Robustesse des conclusions face aux variables de confusion.'
                ]
            }
        },
        {
            id: 'quantum-ai',
            category: 'IA Responsable & Avancée',
            title: '44. IA Quantique',
            shortDesc: 'Domaine émergent explorant l\'utilisation des principes de la mécanique quantique pour améliorer les algorithmes d\'IA.',
            details: {
                useCases: [
                    'Optimisation de problèmes complexes (recherche opérationnelle).',
                    'Amélioration de la recherche d\'embeddings et de la reconnaissance de motifs.',
                    'Développement de nouvelles méthodes d\'apprentissage machine.'
                ],
                projectExamples: 'IBM Qiskit pour la recherche en IA quantique, collaborations entre Google et des instituts de recherche.',
                rex: 'Actuellement à un stade de recherche, les ordinateurs quantiques sont limités. Nécessite une expertise en physique quantique et en informatique.',
                goodPractices: [
                    'Suivre les avancées de la recherche en informatique quantique.',
                    'Expérimenter avec des simulateurs quantiques pour comprendre les potentiels.'
                ],
                metrics: [
                    'Gain de performance algorithmique sur certains problèmes spécifiques.',
                    'Réduction du temps de calcul pour des tâches complexes.'
                ]
            }
        }
    ];

    const conceptsContainer = document.getElementById('concepts-container');
    const searchInput = document.getElementById('searchInput');
    const searchButton = document.getElementById('searchButton');

    // Fonction pour regrouper les concepts par catégorie
    function groupConceptsByCategory(conceptsList) {
        const categories = {};
        conceptsList.forEach(concept => {
            if (!categories[concept.category]) {
                categories[concept.category] = [];
            }
            categories[concept.category].push(concept);
        });
        return categories;
    }

    // Fonction pour générer et afficher les cartes de concepts
    function displayConcepts(filteredConcepts = concepts) {
        conceptsContainer.innerHTML = ''; // Nettoyer le contenu existant

        const grouped = groupConceptsByCategory(filteredConcepts);
        
        for (const category in grouped) {
            const categorySection = document.createElement('div');
            categorySection.className = 'category-section mb-5';
            categorySection.innerHTML = `<h3 class="category-title animate__animated animate__fadeIn">${category}</h3>`;

            const rowDiv = document.createElement('div');
            rowDiv.className = 'row row-cols-1 row-cols-md-2 row-cols-lg-3 g-4';

            grouped[category].forEach(concept => {
                const colDiv = document.createElement('div');
                colDiv.className = 'col animate__animated animate__fadeInUp';
                colDiv.setAttribute('data-aos', 'fade-up');

                colDiv.innerHTML = `
                    <div class="card h-100 concept-card shadow-sm">
                        <div class="card-body">
                            <h5 class="card-title text-primary"><i class="fas fa-microchip me-2"></i>${concept.title}</h5>
                            <p class="card-text">${concept.shortDesc}</p>
                            <a href="#" class="btn btn-sm btn-outline-primary mt-3 read-more" data-bs-toggle="collapse" data-bs-target="#${concept.id}Details" aria-expanded="false" aria-controls="${concept.id}Details">En Savoir Plus <i class="fas fa-plus-circle ms-1"></i></a>
                            <div class="collapse mt-3" id="${concept.id}Details">
                                <h6 class="mt-3 text-secondary">Cas d'Usage & Exemples:</h6>
                                <ul class="list-unstyled">
                                    ${concept.details.useCases.map(uc => `<li><i class="fas fa-caret-right me-2 text-info"></i>${uc}</li>`).join('')}
                                </ul>
                                <h6 class="text-secondary">Projets Exemplaires:</h6>
                                <p class="small">${concept.details.projectExamples}</p>
                                <h6 class="text-secondary">REX (Retour d'Expérience):</h6>
                                <p class="small">${concept.details.rex}</p>
                                <h6 class="text-secondary">Bonnes Pratiques:</h6>
                                <ul class="list-unstyled">
                                    ${concept.details.goodPractices.map(gp => `<li><i class="fas fa-check-circle me-2 text-success"></i>${gp}</li>`).join('')}
                                </ul>
                                <h6 class="text-secondary">Métriques:</h6>
                                <ul class="list-unstyled">
                                    ${concept.details.metrics.map(m => `<li><i class="fas fa-chart-line me-2 text-warning"></i>${m}</li>`).join('')}
                                </ul>
                            </div>
                        </div>
                    </div>
                `;
                rowDiv.appendChild(colDiv);
            });
            categorySection.appendChild(rowDiv);
            conceptsContainer.appendChild(categorySection);
        }

        // Réattacher les écouteurs d'événements aux nouveaux boutons "En Savoir Plus"
        attachReadMoreListeners();
    }

    // Fonction de recherche
    function performSearch() {
        const searchTerm = searchInput.value.toLowerCase();
        const filteredConcepts = concepts.filter(concept => {
            return (
                concept.title.toLowerCase().includes(searchTerm) ||
                concept.shortDesc.toLowerCase().includes(searchTerm) ||
                concept.category.toLowerCase().includes(searchTerm) ||
                concept.details.useCases.some(uc => uc.toLowerCase().includes(searchTerm)) ||
                concept.details.projectExamples.toLowerCase().includes(searchTerm) ||
                concept.details.rex.toLowerCase().includes(searchTerm) ||
                concept.details.goodPractices.some(gp => gp.toLowerCase().includes(searchTerm))
            );
        });
        displayConcepts(filteredConcepts);
    }

    // Écouteurs d'événements pour la recherche
    searchButton.addEventListener('click', performSearch);
    searchInput.addEventListener('keyup', (event) => {
        if (event.key === 'Enter') {
            performSearch();
        } else if (searchInput.value === '') { // Afficher tous les concepts si la recherche est vide
            displayConcepts();
        }
    });

    // Effet de défilement pour la navbar
    const navbar = document.querySelector('.navbar');
    window.addEventListener('scroll', () => {
        if (window.scrollY > 50) {
            navbar.classList.add('scrolled');
        } else {
            navbar.classList.remove('scrolled');
        }
    });

    // Smooth scrolling pour les liens de navigation
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            document.querySelector(this.getAttribute('href')).scrollIntoView({
                behavior: 'smooth'
            });
        });
    });

    // Fonction pour attacher les écouteurs aux boutons "En Savoir Plus"
    function attachReadMoreListeners() {
        document.querySelectorAll('.read-more').forEach(button => {
            // S'assurer de ne pas attacher le listener plusieurs fois
            button.removeEventListener('click', toggleReadMoreIcon);
            button.addEventListener('click', toggleReadMoreIcon);
        });
    }

    function toggleReadMoreIcon() {
        const icon = this.querySelector('i');
        if (icon.classList.contains('fa-plus-circle')) {
            icon.classList.remove('fa-plus-circle');
            icon.classList.add('fa-minus-circle');
        } else {
            icon.classList.remove('fa-minus-circle');
            icon.classList.add('fa-plus-circle');
        }
    }

    // Afficher tous les concepts au chargement initial
    displayConcepts();
});