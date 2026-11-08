import unittest
from congoroad.core import DensityAlert, Detection, Settings, in_zone


class CoreTests(unittest.TestCase):
    def test_zone_geometry_and_bottom_anchor(self):
        settings = Settings()
        self.assertTrue(in_zone((500, 800), 1000, 1000, settings))
        self.assertFalse(in_zone((10, 500), 1000, 1000, settings))
        self.assertFalse(in_zone((500, 400), 1000, 1000, settings))
        self.assertFalse(in_zone((500, 1000), 1000, 1000, settings))
        self.assertEqual(Detection((100, 100, 300, 500), "Voiture", .9).anchor, (200, 500))

    def test_alert_uses_video_time_and_resets(self):
        alert = DensityAlert(Settings(threshold=3, persistence=.5))
        self.assertFalse(alert.update(3, 10))
        self.assertFalse(alert.update(4, 10.4))
        self.assertTrue(alert.update(3, 10.5))
        self.assertFalse(alert.update(2, 10.6))
        self.assertFalse(alert.update(3, 10.7))
        self.assertTrue(alert.update(3, 11.2))

    def test_invalid_settings(self):
        for settings in [Settings(threshold=0), Settings(confidence=2), Settings(horizon=1),
                         Settings(top_width=.9, bottom_width=.5), Settings(alpr_every=0),
                         Settings(persistence=float("nan"))]:
            with self.assertRaises(ValueError):
                settings.validate()


if __name__ == "__main__":
    unittest.main()
