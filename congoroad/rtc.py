"""Server-side TURN credential exchange; never send the API token to WebRTC."""
from urllib.parse import quote

import requests


def has_turn(configuration):
    for server in configuration.get("iceServers", []):
        urls = server.get("urls", [])
        if isinstance(urls, str):
            urls = [urls]
        if any(url.startswith(("turn:", "turns:")) for url in urls):
            return True
    return False


def build_rtc_configuration(secrets):
    options = secrets.get("webrtc", {})
    key = options.get("cloudflare_turn_key_id")
    token = options.get("cloudflare_turn_api_token")
    if key or token:
        if not key or not token:
            raise ValueError("La configuration Cloudflare TURN est incomplète.")
        try:
            response = requests.post(
                "https://rtc.live.cloudflare.com/v1/turn/keys/"
                + quote(str(key), safe="") + "/credentials/generate-ice-servers",
                headers={"Authorization": f"Bearer {token}"},
                json={"ttl": 3600}, timeout=15,
            )
            response.raise_for_status()
            servers = response.json()["iceServers"]
            # Only short-lived ICE credentials leave this function.
            configuration = {"iceServers": [
                {k: server[k] for k in ("urls", "username", "credential") if k in server}
                for server in servers
            ]}
            if not has_turn(configuration):
                raise ValueError("Missing TURN server")
        except (requests.RequestException, ValueError, KeyError, TypeError, AttributeError):
            raise ValueError("Le relais Cloudflare est indisponible. Vérifiez les secrets TURN et réessayez.") from None
    else:
        configuration = {"iceServers": [dict(s) for s in options.get("ice_servers", [])]
                         or [{"urls": ["stun:stun.l.google.com:19302"]}]}
    if options.get("relay_only", False):
        if not has_turn(configuration):
            raise ValueError("Le mode relais nécessite un serveur TURN configuré.")
        configuration["iceTransportPolicy"] = "relay"
    return configuration
