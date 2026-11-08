import unittest
from unittest.mock import Mock, patch

import requests

from congoroad.rtc import build_rtc_configuration, has_turn


class RtcTests(unittest.TestCase):
    def test_local_fallback_and_relay_validation(self):
        self.assertFalse(has_turn(build_rtc_configuration({})))
        with self.assertRaisesRegex(ValueError, "nécessite"):
            build_rtc_configuration({"webrtc": {"relay_only": True}})

    def test_static_turn(self):
        result = build_rtc_configuration({"webrtc": {
            "ice_servers": [{"urls": "turns:example.org:443", "username": "u", "credential": "c"}],
            "relay_only": True}})
        self.assertTrue(has_turn(result))
        self.assertEqual(result["iceTransportPolicy"], "relay")

    @patch("congoroad.rtc.requests.post")
    def test_cloudflare_uses_temporary_credentials_only(self, post):
        post.return_value = Mock(json=lambda: {"iceServers": [{
            "urls": ["turns:turn.cloudflare.com:443?transport=tcp"],
            "username": "temporary-user", "credential": "temporary-password",
            "internal": "do-not-forward"}]})
        result = build_rtc_configuration({"webrtc": {
            "cloudflare_turn_key_id": "key", "cloudflare_turn_api_token": "secret-token"}})
        self.assertTrue(has_turn(result))
        self.assertNotIn("secret-token", str(result))
        self.assertNotIn("internal", str(result))
        self.assertEqual(post.call_args.kwargs["json"], {"ttl": 3600})
        self.assertEqual(post.call_args.kwargs["timeout"], 15)

    @patch("congoroad.rtc.requests.post", side_effect=requests.Timeout("secret-token"))
    def test_provider_errors_do_not_expose_secrets(self, post):
        with self.assertRaises(ValueError) as caught:
            build_rtc_configuration({"webrtc": {
                "cloudflare_turn_key_id": "key", "cloudflare_turn_api_token": "secret-token"}})
        self.assertNotIn("secret-token", str(caught.exception))

    def test_partial_credentials_do_not_silently_fall_back(self):
        with self.assertRaisesRegex(ValueError, "incomplète"):
            build_rtc_configuration({"webrtc": {"cloudflare_turn_key_id": "key"}})
