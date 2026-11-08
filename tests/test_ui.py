import tkinter as tk
import unittest
from unittest.mock import patch

from app import Application


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.root = tk.Tk()
        self.root.withdraw()
        self.app = Application(self.root)

    def tearDown(self):
        # Dispose scheduled callbacks before destroying the Tcl interpreter.
        for callback in self.root.tk.call("after", "info"):
            self.root.after_cancel(callback)
        self.root.destroy()

    def test_preview_queue_keeps_only_latest_frame(self):
        for index in range(100):
            self.app.emit("frame", {"current": index})
        self.assertEqual(self.app.latest.qsize(), 1)
        self.assertEqual(self.app.latest.get_nowait()["current"], 99)

    def test_invalid_settings_do_not_start_worker(self):
        self.app.threshold.set("0")
        with patch("app.messagebox.showerror") as error:
            self.app.start()
        error.assert_called_once()
        self.assertIsNone(self.app.worker)
        self.assertNotIn("disabled", self.app.run_button.state())


if __name__ == "__main__":
    unittest.main()
