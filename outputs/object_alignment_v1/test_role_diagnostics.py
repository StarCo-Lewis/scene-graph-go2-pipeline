import unittest
from align_objects import validate_decomposition, AlignmentError


class RoleDiagnosticTests(unittest.TestCase):
    def test_missing_role_id_reports_exact_path_and_keys(self):
        data = {'subtasks': [{'subtask_id': 'st_01', 'required_roles': [{'id': 'kettle'}]}]}
        with self.assertRaisesRegex(AlignmentError, r'required_roles\[1\].role_id.*available keys'):
            validate_decomposition(data)

    def test_wrong_role_id_types_rejected_not_coerced(self):
        for value in (None, 67, [], ''):
            with self.subTest(value=value), self.assertRaises(AlignmentError):
                validate_decomposition({'subtasks': [{'subtask_id': 'st_01',
                    'required_roles': [{'role_id': value}]}]})
