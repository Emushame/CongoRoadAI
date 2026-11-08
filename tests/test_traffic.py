import unittest

from congoroad.core import Detection, Settings
from congoroad.traffic import TrafficMonitor


def vehicles(n=3):
    return [Detection((180+i*80, 240, 250+i*80, 450), 'Voiture', .9) for i in range(n)]


class TrafficTests(unittest.TestCase):
    def run_scene(self, monitor, seconds=12, fps=10, count=3):
        for i in range(int(seconds*fps)+1):
            result = monitor.update(vehicles(count), 640, 480, i/fps)
        return result

    def test_warmup_and_sustained_density(self):
        monitor = TrafficMonitor(Settings())
        result = self.run_scene(monitor, 5)
        self.assertEqual(result['level'], 'observing')
        for i in range(51, 121):
            result = monitor.update(vehicles(), 640, 480, i/10)
        self.assertEqual(result['level'], 'risk')
        self.assertEqual(result['tracked'], 3)
        self.assertIsNone(result['low_motion'])

    def test_empty_scene_does_not_claim_free_road(self):
        result = self.run_scene(TrafficMonitor(Settings()), count=0)
        self.assertEqual(result['level'], 'none')
        self.assertIn('ne garantit pas', result['reasons'][0])

    def test_transient_density_is_not_risk(self):
        monitor = TrafficMonitor(Settings())
        self.run_scene(monitor, count=0)
        for i in range(121, 151):
            result = monitor.update(vehicles(),640,480,i/10)
        self.assertNotEqual(result['level'], 'risk')

    def test_accumulation_before_dense_threshold(self):
        monitor = TrafficMonitor(Settings(threshold=5))
        for i in range(141):
            n = 1 if i < 70 else (2 if i < 100 else 3)
            result = monitor.update(vehicles(n),640,480,i/10)
        self.assertEqual(result['level'], 'watch')

    def test_gap_and_rewind_reset_evidence(self):
        for timestamp in (0, 20):
            monitor = TrafficMonitor(Settings())
            self.run_scene(monitor)
            result = monitor.update(vehicles(),640,480,timestamp)
            self.assertEqual(result['level'], 'observing')
            self.assertEqual(result['tracked'], 0)

    def test_fixed_camera_motion_and_sampling_invariance(self):
        for fps in (2, 10, 30):
            result = self.run_scene(TrafficMonitor(Settings(camera_fixed=True)), fps=fps)
            self.assertEqual(result['low_motion'], 3)
            self.assertEqual(result['level'], 'risk')

    def test_occupancy_is_union_clipped_to_zone(self):
        monitor = TrafficMonitor(Settings())
        box = Detection((-100,-100,1000,1000),'Voiture',.9)
        result = monitor.update([box,box],640,480,0)
        self.assertEqual(result['occupancy'], 1)

    def test_monitors_are_isolated_and_history_bounded(self):
        first, second = TrafficMonitor(Settings()), TrafficMonitor(Settings())
        self.run_scene(first, seconds=100)
        self.assertLessEqual(len(first.history), 61)
        self.assertEqual(second.update(vehicles(),640,480,0)['level'], 'observing')
