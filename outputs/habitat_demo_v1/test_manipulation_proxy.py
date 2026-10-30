import importlib.util
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).parent
for name in ("run_navigation_demo", "run_manipulation_proxy"):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)

proxy = sys.modules["run_manipulation_proxy"]


class ManipulationProxyTests(unittest.TestCase):
    def test_contact_with_kettle_on_board_top(self):
        contact = SimpleNamespace(object_id_a=42, object_id_b=-1,
                                  position_on_a_in_ws=[-5.75, 0.96, -1.5])
        self.assertTrue(proxy.contact_near_target(contact, 42, [-5.8, 0.5, -1.5],
                                                  [0.4, 1.2], 0.95))

    def test_floor_contact_does_not_count_as_sideboard(self):
        contact = SimpleNamespace(object_id_a=42, object_id_b=-1,
                                  position_on_a_in_ws=[-5.75, 0.01, -1.5])
        self.assertFalse(proxy.contact_near_target(contact, 42, [-5.8, 0.5, -1.5],
                                                   [0.4, 1.2], 0.95))

    def test_unrelated_object_contact_does_not_count(self):
        contact = SimpleNamespace(object_id_a=99, object_id_b=-1)
        self.assertFalse(proxy.contact_near_target(contact, 42, [-5.8, 0.5, -1.5],
                                                   [0.4, 1.2], 0.95))

    def test_tail_motion(self):
        self.assertAlmostEqual(proxy.tail_motion_m([[0, 0, 0], [0, 0.01, 0]]), 0.01)


if __name__ == "__main__":
    unittest.main()
