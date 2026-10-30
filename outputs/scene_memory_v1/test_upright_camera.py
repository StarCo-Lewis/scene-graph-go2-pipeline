import math
import unittest
from capture_visual_memory import upright_camera_rotation, scan_direction


class UprightCameraTests(unittest.TestCase):
    def test_scan_preserves_length_height_and_is_not_target_driven(self):
        direction = [.4, -.3, .5]
        for yaw in (-35, 0, 35):
            result = scan_direction(direction, yaw)
            self.assertEqual(result[1], direction[1])
            self.assertAlmostEqual(sum(x*x for x in result), sum(x*x for x in direction))
            recovered = scan_direction(result, -yaw)
            for a, b in zip(recovered, direction):
                self.assertAlmostEqual(a, b)

    def test_looks_forward_with_no_roll(self):
        for direction in ([0, 0, -1], [1, -.6, 1], [-.2, -.3, .4]):
            r = upright_camera_rotation(direction)
            length = math.sqrt(sum(x*x for x in direction))
            for i in range(3):
                self.assertAlmostEqual(-r[i][2], direction[i]/length)
            self.assertAlmostEqual(r[1][0], 0)
            self.assertGreater(r[1][1], 0)
            for a in range(3):
                for b in range(3):
                    self.assertAlmostEqual(sum(r[i][a]*r[i][b] for i in range(3)), float(a == b))
            right, up, backward = ([r[i][j] for i in range(3)] for j in range(3))
            cross = [right[1]*up[2]-right[2]*up[1], right[2]*up[0]-right[0]*up[2], right[0]*up[1]-right[1]*up[0]]
            for a, b in zip(cross, backward):
                self.assertAlmostEqual(a, b)

    def test_degenerate_or_nonfinite_direction_rejected(self):
        for direction in ([0, 0, 0], [0, 1, 0], [float('nan'), 0, 1]):
            with self.assertRaises(ValueError):
                upright_camera_rotation(direction)
