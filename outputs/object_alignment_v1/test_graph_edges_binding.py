import unittest
from llm_graph_binding import validate_edges, check_binding_relations, BindingError


class EdgeTests(unittest.TestCase):
    def setUp(self):
        self.bindings = {'can': {'element_ids': [6]}, 'table': {'element_ids': [2]}}
        self.constraints = [{'constraint_id': 'on_table', 'subject_role': 'can', 'object_role': 'table', 'relation': 'on', 'required': True}]

    def test_matching_edge(self):
        checks, errors = check_binding_relations(self.bindings, self.constraints, [{'source': 6, 'target': 2, 'relation': 'a on b'}])
        self.assertFalse(errors); self.assertEqual(checks[0]['status'], 'observed_true')

    def test_reverse_edge(self):
        self.assertFalse(check_binding_relations(self.bindings, self.constraints, [{'source': 2, 'target': 6, 'relation': 'b on a'}])[1])

    def test_unknown_not_no_object(self):
        self.assertEqual(check_binding_relations(self.bindings, self.constraints, [])[1][0]['reason'], 'relation_unverified')

    def test_conflict(self):
        self.assertEqual(check_binding_relations(self.bindings, self.constraints, [{'source': 6, 'target': 5, 'relation': 'a on b'}])[1][0]['reason'], 'relation_conflict')

    def test_invalid_endpoint(self):
        with self.assertRaises(BindingError):
            validate_edges([{'source': 99, 'target': 2, 'relation': 'a on b'}], [{'id': 2}])
