from dataclasses import asdict
from pathlib import Path
import hashlib
import json
import time

import streamlit as st

from congoroad.core import Settings
from congoroad.live import SharedModels
from congoroad.web_jobs import VideoJob
from congoroad.rtc import build_rtc_configuration, has_turn

st.set_page_config(page_title="CongoRoadAI · Vision de la route", page_icon="🛣️", layout="wide")
st.markdown("""<style>
.block-container {max-width:1240px; padding-top:2.3rem;}
[data-testid="stSidebar"] {border-right:1px solid #23333e;}
.brand {font-size:15px;font-weight:750;letter-spacing:.15em;color:#dce8ec;}
.brand span {color:#45d7ac;}
.eyebrow {color:#45d7ac;font-size:12px;font-weight:650;letter-spacing:.16em;margin:32px 0 10px;}
.hero {font-size:clamp(32px,4.5vw,56px);font-weight:750;letter-spacing:-.04em;line-height:1.08;margin:0 0 18px;}
.intro {color:#a9bac5;max-width:750px;font-size:17px;line-height:1.65;margin-bottom:24px;}
.empty {border:1px dashed #3b5362;border-radius:18px;padding:60px 24px;text-align:center;background:linear-gradient(140deg,#142b31,#101c29);}
.empty b {display:block;font-size:23px;color:#e5f3ef;margin:12px;}
.empty p {color:#9bb3be;max-width:520px;margin:auto;line-height:1.7;}
.legend {color:#a7b8bf;font-size:13px;margin:16px 0 24px;}
.footer {border-top:1px solid #273541;padding-top:18px;color:#93a6b2;font-size:12px;margin-top:36px;}
[data-testid="stMetric"] {background:#13232e;border:1px solid #29404b;border-radius:12px;padding:14px;}
</style>""", unsafe_allow_html=True)


@st.cache_resource(max_entries=2, show_spinner=False)
def get_models(with_alpr):
    return SharedModels(with_alpr)


def rtc_configuration():
    try:
        options = dict(st.secrets.get("webrtc", {}))
    except FileNotFoundError:
        options = {}
    return build_rtc_configuration({"webrtc": options})


st.markdown('<div class="brand">CONGO<span>ROAD</span>AI <span> / </span> LAB</div>', unsafe_allow_html=True)
st.markdown('<div class="eyebrow">VISION PAR ORDINATEUR · PROTOTYPE EXPÉRIMENTAL</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Un autre regard<br>sur la route.</div>', unsafe_allow_html=True)
st.markdown('<div class="intro">Repérez les véhicules, observez les zones de circulation et explorez la lecture des plaques. Depuis une vidéo ou votre caméra.</div>', unsafe_allow_html=True)

with st.sidebar:
    st.title("Votre analyse")
    mode = st.radio("Source", ["Vidéo importée", "Caméra en direct"], key="source_mode")
    st.divider()
    st.subheader("Détection")
    confidence = st.slider("Confiance YOLO", .10, .90, .35, .05)
    threshold = st.slider("Seuil de prudence", 2, 10, 3, help="Nombre de véhicules simultanément dans la zone.")
    persistence = st.slider("Persistance (secondes)", 0.0, 3.0, .5, .1)
    alpr = st.toggle("Lire les plaques", value=False, help="FastALPR demande plus de calcul et peut réduire la fluidité.")
    with st.expander("Zone de circulation"):
        horizon = st.slider("Horizon", .15, .80, .42, .01)
        top_width = st.slider("Largeur en haut", .10, .60, .30, .05)
        bottom_width = st.slider("Largeur en bas", .60, 1.0, .90, .05)
    with st.expander("Performances"):
        width = st.select_slider("Largeur de traitement", options=[480, 640, 960, 1280], value=640)
        alpr_every = st.slider("Lire les plaques toutes les N images", 1, 30, 10)
    st.caption("La zone doit être adaptée au cadrage. Les couleurs distinguent les catégories, pas des identités suivies.")

settings = Settings(confidence=confidence, threshold=threshold, persistence=persistence,
                    horizon=horizon, top_width=top_width, bottom_width=bottom_width,
                    alpr=alpr, alpr_every=alpr_every, max_width=width)
signature = hashlib.sha256(json.dumps(asdict(settings), sort_keys=True).encode()).hexdigest()[:12]
st.markdown('<div class="legend">🟢 Voiture &nbsp; · &nbsp; 🔵 Moto &nbsp; · &nbsp; 🟠 Bus &nbsp; · &nbsp; 🟣 Camion &nbsp; · &nbsp; 🔴 Zone dense</div>', unsafe_allow_html=True)


@st.fragment(run_every=1)
def job_panel():
    job = st.session_state.get("video_job")
    if job is None:
        return
    state = job.snapshot()
    st.subheader(state["status"])
    if not state["done"]:
        if st.button("Arrêter le traitement", key="stop_job"):
            job.stop.set()
            st.info("Arrêt demandé ; l’image en cours sera terminée.")
        st.progress(min(1.0, state.get("current", 0) / max(1, state.get("total", 1))))
        if "image" in state:
            st.image(state["image"], channels="BGR", use_container_width=True)
        st.caption(f"{state.get('current', 0)} / {state.get('total', '?')} images · {state.get('vehicles', 0)} véhicule(s) dans la zone")
    if state.get("warning"):
        st.warning(state["warning"])
    if state.get("error"):
        st.error(state["error"])
    if state["done"]:
        if state.get("summary"):
            summary = state["summary"]
            cols = st.columns(3)
            cols[0].metric("Images analysées", summary["frames"])
            cols[1].metric("Images avec alerte", summary["alert_frames"])
            cols[2].metric("Temps de calcul", f"{summary['processing_seconds']:.1f} s")
        if state.get("video"):
            st.video(state["video"], format="video/mp4")
            st.download_button("Télécharger la vidéo", state["video"], "congoroad-video.mp4", "video/mp4")
        if state.get("archive"):
            st.download_button("Télécharger tous les résultats (ZIP)", state["archive"], "congoroad-resultats.zip", "application/zip")
        if st.button("Effacer les résultats et la vidéo", key="clear_job"):
            job.thread.join(timeout=1)
            job.cleanup()
            del st.session_state["video_job"]
            st.rerun()


if mode == "Vidéo importée":
    st.subheader("01 / Importer une vidéo")
    uploaded = st.file_uploader("Vidéo de trajet", type=["mp4", "mov", "avi", "mkv", "m4v"],
                                help="100 Mo et 2 minutes maximum pour cette démonstration.")
    consent = st.checkbox("J’accepte le traitement de cette vidéo sur le serveur de l’application.", key="upload_consent")
    st.caption("En local, le serveur est votre ordinateur. Sur Streamlit Cloud, les images quittent votre appareil. Les résultats restent temporaires ; utilisez « Effacer » pour les supprimer.")
    existing = st.session_state.get("video_job")
    busy = existing is not None and not existing.snapshot()["done"]
    if st.button("Analyser la vidéo", type="primary", disabled=uploaded is None or not consent or busy):
        try:
            with st.spinner("Préparation des modèles… Le premier lancement peut télécharger leurs poids."):
                models = get_models(alpr)
                if existing is not None:
                    existing.thread.join(timeout=1)
                    existing.cleanup()
                    del st.session_state["video_job"]
                st.session_state["video_job"] = VideoJob(uploaded.getvalue(), Path(uploaded.name).suffix, settings, models)
        except Exception as exc:
            st.error(f"Impossible de démarrer : {exc}")
    if uploaded is not None:
        with st.expander("Lire la vidéo originale"):
            st.video(uploaded.getvalue())
    if st.session_state.get("video_job") is None:
        st.markdown('<div class="empty"><span style="font-size:38px">↗</span><b>Votre prochaine exploration commence ici.</b><p>Importez un court trajet. Retrouvez la vidéo annotée, les alertes et un rapport téléchargeable au même endroit.</p></div>', unsafe_allow_html=True)
    job_panel()
else:
    st.subheader("02 / Observer en direct")
    st.info("La caméra s’ouvre uniquement après votre action. Autorisez-la dans le navigateur ; le microphone reste désactivé.")
    consent = st.checkbox("J’accepte l’envoi du flux caméra au serveur pour analyse.", key="camera_consent")
    facing = st.selectbox("Caméra préférée", ["Arrière / route", "Avant / webcam"])
    prepared = st.session_state.get("camera_prepared") == signature
    if st.button("Préparer la caméra", type="primary", disabled=not consent):
        try:
            with st.spinner("Chargement des modèles pour le direct…"):
                st.session_state["camera_rtc"] = rtc_configuration()
                st.session_state["camera_rtc_created"] = time.monotonic()
                st.session_state["camera_models"] = get_models(alpr)
                st.session_state["camera_prepared"] = signature
            prepared = True
        except Exception as exc:
            st.error(f"La caméra ne peut pas être préparée : {exc}")
    if consent and prepared:
        if time.monotonic() - st.session_state.get("camera_rtc_created", 0) > 3300:
            st.info("La connexion a expiré. Cliquez sur « Préparer la caméra » pour la renouveler.")
            st.stop()
        configuration = st.session_state["camera_rtc"]
        if not has_turn(configuration):
            st.warning("Le relais vidéo n’est pas encore configuré. Sur ce réseau, le direct peut rester bloqué. L’analyse de vidéos importées reste disponible.")
        from streamlit_webrtc import WebRtcMode, webrtc_streamer
        from congoroad.webcam import CameraProcessor
        models = st.session_state["camera_models"]
        context = webrtc_streamer(
            key=f"road-camera-{signature}-{facing}-{st.session_state['camera_rtc_created']}", mode=WebRtcMode.SENDRECV,
            rtc_configuration=configuration,
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480},
                                               "frameRate": {"ideal": 10, "max": 15},
                                               "facingMode": {"ideal": "environment" if facing == "Arrière / route" else "user"}},
                                      "audio": False},
            video_processor_factory=lambda: CameraProcessor(settings, models), async_processing=True,
            video_html_attrs={"autoPlay": True, "controls": True, "muted": True, "playsInline": True, "style": {"width": "100%"}},
        )

        @st.fragment(run_every=1)
        def camera_metrics():
            if not context.state.playing or context.video_processor is None:
                st.caption("Appuyez sur START pour ouvrir la caméra, puis STOP pour la fermer.")
                return
            snapshot = context.video_processor.snapshot()
            if snapshot.get("error"):
                st.error("Erreur d’analyse : " + snapshot["error"])
            elif time.monotonic() - snapshot.get("updated_at", 0) > 3:
                st.info("En attente d’images récentes…")
            else:
                a, b, c = st.columns(3)
                a.metric("Véhicules dans la zone", snapshot.get("vehicles", 0))
                b.metric("Temps par image", f"{snapshot.get('latency_ms', 0)} ms")
                c.metric("Images traitées", snapshot.get("processed", 0))
                if snapshot.get("alert"):
                    st.warning("Prudence indicative : plusieurs véhicules occupent la zone.")
        camera_metrics()
    elif st.session_state.get("camera_prepared") and not prepared:
        st.caption("Les réglages ont changé. Préparez à nouveau la caméra pour les appliquer.")
    st.caption("Le flux direct n’est pas enregistré par l’application. Le traitement peut sauter des images pour limiter le retard ; le temps réel n’est pas garanti sur un serveur partagé.")
    with st.expander("La caméra ne se connecte pas ?"):
        st.write("Utilisez Chrome ou Edge, autorisez la caméra et ouvrez l’application en HTTPS (ou localhost). Vérifiez qu’aucune autre application n’occupe la caméra. Sur certains réseaux et sur Streamlit Cloud, un relais TURN configuré par l’hébergeur est nécessaire.")

st.markdown('<div class="footer">CongoRoadAI · Projet expérimental. Les alertes ne constituent pas un dispositif de sécurité et ne remplacent pas l’attention du conducteur. Les lectures de plaques peuvent être erronées.</div>', unsafe_allow_html=True)
