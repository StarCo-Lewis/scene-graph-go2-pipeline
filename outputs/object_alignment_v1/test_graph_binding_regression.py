import unittest
from run_graph_binding_regression import cases


class RegressionFixtureTests(unittest.TestCase):
    def test_same_descriptions_distinct_ids(self):
        nodes = [{'id': 4, 'object_tag': 'coke can', 'caption': 'Red can', 'bbox_center': [2, 0, 1]}]
        fixtures = cases(nodes)
        self.assertEqual(len(nodes), 1)
        cans = fixtures[1][1]
        self.assertEqual(cans[0]['caption'], cans[1]['caption'])
        self.assertNotEqual(cans[0]['id'], cans[1]['id'])
        self.assertNotEqual(cans[0]['bbox_center'], cans[1]['bbox_center'])
        self.assertEqual(fixtures[0][3], 'NO_OBJECT_CAN_BE_USED_IN_THIS_TASK')
