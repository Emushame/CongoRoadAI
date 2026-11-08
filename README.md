# CongoRoadAI

**Application publique : https://congoroad-ai.streamlit.app/**  
**Code source : https://github.com/Emushame/CongoRoadAI**

Déploiement vérifié le 8 novembre 2026 sur Streamlit Community Cloud, Python 3.11 : une vidéo synthétique de 20 images a été analysée avec YOLO et FastALPR en 6,1 secondes, avec les exports MP4 et ZIP disponibles. Ce test valide le fonctionnement du pipeline hébergé, pas la précision sur route. Le test caméra distant est resté en attente de connexion avec le STUN seul, puis a été arrêté : un service TURN reste à configurer pour ce réseau. La caméra a été validée en exécution locale.

Application en français pour analyser une vidéo ou une **caméra en direct**, avec Python, OpenCV, YOLO et FastALPR. Deux interfaces : bureau Tkinter et web Streamlit. Traitement sur CPU. Avec la version web publiée, les images sont envoyées au serveur Streamlit pour analyse ; la version exécutée localement reste sur votre ordinateur.

## Application web et caméra

Après installation des dépendances, lancer `lancer_web.bat` ou :

```powershell
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Ouvrir l'adresse locale indiquée. **Vidéo importée** permet de lire l'original, d'analyser un extrait de 100 Mo / 2 minutes maximum et de télécharger une vidéo H.264 lisible dans le navigateur ainsi qu'un ZIP avec MP4, CSV et JSON. Un traitement est limité à 5 minutes de calcul, puis finalise un export partiel. Le bouton d'arrêt agit après l'image en cours. Les fichiers sont propres à chaque session et supprimables avec « Effacer les résultats et la vidéo » ; ils restent temporaires pendant la session. L'effacement des résultats ne retire pas le fichier déjà sélectionné dans le widget d'import : utiliser sa croix pour le retirer aussi.

**Caméra en direct** utilise WebRTC pour accéder à la caméra du navigateur (pas à celle du serveur). Accepter le traitement, cliquer sur « Préparer la caméra », puis START et autoriser la caméra. STOP termine le flux. Le microphone est désactivé et aucun enregistrement du direct n'est créé. Les images anciennes sont abandonnées lorsque l'inférence prend du retard. La cadence de traitement n'est pas garantie, en particulier avec l'OCR et sur un serveur partagé. Une coupure du flux réinitialise la temporisation de prudence.

## Publication sur Streamlit Community Cloud

### Relais Cloudflare pour la caméra distante

Créer une clé TURN dans Cloudflare Realtime, puis saisir dans **Streamlit > App settings > Secrets** (jamais dans GitHub) :

```toml
[webrtc]
cloudflare_turn_key_id = "identifiant-de-la-cle-TURN"
cloudflare_turn_api_token = "jeton-de-la-cle-TURN"
relay_only = true
```

Le serveur échange ces secrets contre des identifiants temporaires valables une heure, propres à chaque préparation de caméra. Seuls ces identifiants temporaires sont transmis au navigateur. Cliquer sur « Préparer la caméra » après configuration ou expiration. Aucun compte ni secret réel Cloudflare n’est fourni dans le dépôt. Une nouvelle validation du direct distant reste nécessaire après ajout des secrets. [Procédure officielle Cloudflare](https://developers.cloudflare.com/realtime/turn/generate-credentials/).

### Déploiement

Le point d'entrée est **`streamlit_app.py`**, pas `app.py` qui démarre Tkinter. Le dépôt contient `requirements.txt`, `packages.txt` pour les bibliothèques Linux et `.streamlit/config.toml`. Python **3.11** est la version testée. PyTorch CPU est sélectionné pour Linux ; le verrou `requirements-lock.txt` décrit l'environnement Windows et n'est pas le fichier à sélectionner pour déployer sur Linux.

1. Publier le code dans un dépôt GitHub, en excluant `.venv`, `data`, `outputs`, les vidéos, les poids et les secrets (déjà ignorés par Git).
2. Dans [Streamlit Community Cloud](https://share.streamlit.io/), créer une application depuis ce dépôt, sélectionner la branche publiée, `streamlit_app.py` et Python 3.11 dans les paramètres avancés.
3. Pour un direct fiable sur les réseaux filtrés, configurer un **serveur TURN** dans les secrets Streamlit, avec la structure de `.streamlit/secrets.example.toml`. Le STUN public par défaut peut suffire en local mais ne garantit pas une connexion à travers tous les NAT/pare-feux. Les identifiants ICE sont transmis au navigateur : utiliser un service dédié et privilégier des identifiants temporaires. Ne pas committer de vraies clés.
4. Déployer puis tester l'URL HTTPS depuis un autre appareil, autorisation caméra incluse. Les modèles sont téléchargés lors de leur première utilisation et mis en cache côté serveur ; prévoir un premier lancement plus lent.

La publication nécessite une session GitHub/Streamlit autorisée. Une page qui fonctionne en local ne constitue pas une validation d'hébergement ni du réseau WebRTC distant. Références : [déploiement Streamlit](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy), [streamlit-webrtc](https://github.com/whitphx/streamlit-webrtc).

## Démarrer sous Windows

Python 3.11 recommandé. Double-cliquer sur `installer.bat`, ou dans PowerShell, depuis ce dossier :

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe app.py
```

Une fois installé, double-cliquer sur `lancer.bat`. Choisir une vidéo, régler les paramètres puis cliquer sur **Analyser et exporter**. L'option FastALPR active la détection et la lecture des plaques. Le bouton **Arrêter** conserve un export partiel lisible ; il prend effet après l'opération d'inférence en cours.

Les modèles sont téléchargés au premier usage : une connexion Internet est nécessaire à cette étape. Les inférences suivantes utilisent les modèles en cache. La vidéo n'est pas envoyée à une API. Un échec de téléchargement ou de FastALPR est signalé explicitement.

## Fonctionnement

- YOLO11n préentraîné sur COCO détecte les voitures, motos, bus et camions ; chaque catégorie a sa couleur. Pas de suivi d'identités entre les images.
- Une zone trapézoïdale représente la chaussée devant la caméra. Un véhicule est compté lorsque le milieu du bas de sa boîte se trouve dans cette zone.
- Par défaut, au moins **3 véhicules pendant 0,5 seconde de vidéo** déclenchent l'alerte. La flèche rouge pointe vers les véhicules du tiers horizontal le plus peuplé. Elle ne recommande aucune manœuvre.
- FastALPR travaille toutes les 5 images ; une lecture sous 70 % de confiance est marquée « Illisible ». Les résultats sont affichés uniquement sur l'image analysée pour éviter de déplacer une ancienne plaque sur un autre véhicule.
- Les images sont réduites à 1280 pixels de large au maximum. L'export conserve la cadence d'origine, mais **sans audio**. L'aperçu avance à la vitesse du calcul, sans garantie de temps réel.

L'horizon correspond à la hauteur relative du haut de la zone : `0.42` signifie 42 % de la hauteur de l'image. Les largeurs de la zone sont réglables en ligne de commande. La zone doit être adaptée au cadrage de chaque vidéo ; elle ne détecte pas automatiquement les voies.

## Exports

Chaque analyse crée un nouveau dossier dans `outputs/` contenant :

- `video_annotee.mp4` : vidéo avec les annotations ;
- `analyse.csv` : une ligne par image, temps, détections, véhicules dans la zone, alerte, exécution OCR et plaques JSON ;
- `resume.json` : paramètres, durée, nombre d'images et indicateur d'arrêt anticipé.

En cas d'erreur, `erreur.txt` indique que les exports sont incomplets. Le nombre de détections par image n'est pas un nombre de véhicules uniques. Les plaques reconnues sont des hypothèses OCR et restent dans les exports locaux.

## Vidéos publiques pour commencer

Deux sources Kaggle à considérer :

1. [Road Traffic Video Monitoring — shawon10](https://www.kaggle.com/datasets/shawon10/road-traffic-video-monitoring). Le [notebook du même auteur](https://www.kaggle.com/code/shawon10/object-detection-from-a-traffic-video/input) répertorie `road_trafifc.mp4`, `traffic_detection.mp4` et `traffic_video.avi`, pour environ 26 Mo au total. Commencer par une vidéo brute pour vérifier les boîtes et les exports. Ce choix ne garantit pas le point de vue d'une caméra embarquée.
2. [Nexar Dashcam Crash Prediction](https://www.kaggle.com/competitions/nexar-collision-prediction/data). Les vidéos MP4 sont filmées vers la route par des dashcams : le cadrage convient mieux à l'analyse devant le véhicule. Consulter les conditions de la compétition et les modalités d'accès sur Kaggle.

Télécharger et extraire une vidéo dans `data/videos/` (dossier ignoré par Git), puis la sélectionner dans l'application. Un compte Kaggle peut être requis selon la source. Commencer avec FastALPR désactivé, ajuster la zone, puis activer l'OCR sur une vidéo où les plaques sont assez grandes. Ces sources servent d'abord à tester le pipeline et les véhicules ; leur aptitude à évaluer les plaques n'a pas été vérifiée. Aucun de ces fichiers n'est inclus ou présenté comme déjà testé.

## Ligne de commande

```powershell
.\.venv\Scripts\python.exe -m congoroad "C:\Videos\trajet.mp4" --alpr --threshold 3
.\.venv\Scripts\python.exe -m congoroad "C:\Videos\trajet.mp4" --horizon 0.45 --top-width 0.25 --bottom-width 0.85 --persistence 0.8
```

`--output` accepte un **nouveau** dossier ; les dossiers existants ne sont jamais écrasés. `--model` accepte un modèle YOLO compatible avec les classes COCO utilisées. `--alpr-every 1` lit les plaques sur chaque image, au prix d'un calcul plus lent.

## Vérification

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Les tests du pipeline utilisent une vidéo synthétique et un détecteur simulé pour vérifier les exports, la temporisation, l'arrêt et les erreurs. Ils ne mesurent pas la précision de YOLO ou de l'OCR sur une route réelle.

Vérification du 8 novembre 2026 : **21 tests automatisés réussis** sous Python 3.11, avec OpenCV 4.11.0.86, NumPy 2.4.6 et Pillow 12.3.0. Ils couvrent la géométrie, la temporisation, les paramètres invalides, la relecture intégrale du MP4, le CSV, l'arrêt volontaire, une panne du détecteur, une vidéo invalide, une source absente, les lectures OCR incertaines, l'intervalle OCR et la file d'aperçu de l'interface. Les tests vidéo utilisent des détecteurs simulés ; ils ne valident pas les modèles réels. Les tests web couvrent aussi l’isolation des sessions, les coupures temporelles, l’abandon des anciennes images, les erreurs caméra, les limites d’import, l’export H.264 et le consentement de l’interface Streamlit. Un essai séparé avec les modèles réels a produit un MP4 H.264 et un ZIP à partir de 20 images synthétiques. La caméra du navigateur a également transmis des images traitées en local, puis a été arrêtée ; le réseau WebRTC distant reste à valider après déploiement.

Toutes les dépendances sont installées : `pip check` ne signale aucun conflit et les imports de YOLO, FastALPR, PyTorch 2.15.0 CPU et ONNX Runtime 1.31.1 réussissent. Les versions exactes de cet environnement Windows/Python 3.11 sont enregistrées dans `requirements-lock.txt` ; utiliser `pip install -r requirements-lock.txt` pour les reproduire.

Le test séparé des modèles se lance explicitement (téléchargement possible des poids) :

```powershell
.\.venv\Scripts\python.exe -X utf8 -m scripts.check_models
```

Il consigne son résultat dans `outputs/validation/model-smoke.json`. Le 8 novembre 2026, le chargement des modèles et l'inférence sur une image synthétique ont réussi pour YOLO et FastALPR. Les poids YOLO, du détecteur de plaques et de l'OCR sont téléchargés et disponibles localement. L'image vide ne comporte aucune plaque : ce contrôle ne mesure ni la qualité des détections ni la précision de lecture OCR. La validation sur vidéo réelle reste à effectuer.

## Limites

**Projet expérimental, pas un dispositif de sécurité ni un substitut à l'attention du conducteur.** Le nombre d'objets dans une zone est un indicateur simple, pas une mesure de distance, de risque de collision ou de densité physique. Les plaques petites, floues, masquées ou de formats locaux peuvent être mal lues. Les modèles ne sont pas spécialement entraînés sur les routes ou les plaques congolaises. Aucune validation de précision sur une vidéo de trajet n'est incluse dans ce dépôt.

Références des intégrations : [Ultralytics](https://docs.ultralytics.com/modes/predict/) et [FastALPR](https://ankandrew.github.io/fast-alpr/latest/reference/). Vérifier les licences des bibliothèques et des modèles avant une redistribution commerciale.
