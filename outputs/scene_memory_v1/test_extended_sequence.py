import unittest
from unittest.mock import patch
from run_visual_sequence import sequence_moves, evidence_guards
from update_memory import initialize, digest


class ExtendedSequenceTests(unittest.TestCase):
    def test_five_moves_bounded_and_back_compatible(self):
        self.assertEqual(sequence_moves(2), [[.15, 0, 0], [-.2, 0, .1]])
        self.assertEqual(len(sequence_moves(5)), 5)
        self.assertEqual(sequence_moves(2, 3), [[-.15, 0, .1], [.1, 0, -.1]])
        with self.assertRaises(ValueError):
            sequence_moves(5, 3)
        position = [0., 0., 0.]
        for delta in sequence_moves(6):
            position = [a+b for a, b in zip(position, delta)]
            self.assertLessEqual(sum(x*x for x in position)**.5, .2)
        for count in (1, 7):
            with self.assertRaises(ValueError):
                sequence_moves(count)

    def test_guard_checks_leave_input_untouched(self):
        state = initialize([{'id': 67, 'bbox_center': [1, 2, 3]}], [], 'fixture')
        before = digest(state)
        estimate = {'views': [{'translation_cg': [.25, 0, 0], 'symmetric_trimmed_rms_m': .01},
                              {'translation_cg': [.251, 0, 0], 'symmetric_trimmed_rms_m': .01}]}
        from pathlib import Path
        with patch('run_visual_sequence.read', return_value=estimate):
            report = evidence_guards(Path('previous'), state)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(len(report['checks']), 5)
        self.assertEqual(digest(state), before)
