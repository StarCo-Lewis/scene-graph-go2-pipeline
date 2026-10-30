import copy
import json
from pathlib import Path
import tempfile
import unittest
import test_action_matrix_ab as fixtures
from action_matrix import matrix_payload, decode_matrix
from match_skills import PlanError
from matrix_type_contract import MatrixTypeError
from skill_output_normalization import normalize, normalize_and_record


class SkillMatrixTypePolicyTests(unittest.TestCase):
    def test_skill_generation_uses_typed_tuple_and_current_codes(self):
        d,lib,state,book,matrix,retrieval=fixtures.inputs()
        payload=matrix_payload(d['alignment'],lib,book,'offline',retrieval,False,matrix_contract='strict_schema')
        fmt=payload['response_format']; self.assertEqual(fmt['type'],'json_schema')
        row=fmt['json_schema']['schema']['properties']['matrix']['items']
        self.assertEqual([p['type'] for p in row['prefixItems']],['integer']*4)
        self.assertEqual(row['prefixItems'][0]['enum'],[1,2])

    def test_compatibility_preserves_actions_and_binding_meaning(self):
        d,lib,state,book,matrix,_=fixtures.inputs()
        raw={'matrix':[[str(c) for c in row] for row in matrix['matrix']]}
        before=copy.deepcopy(raw)
        value,record=normalize(raw,'matrix',lib,matrix_contract='typed_json')
        self.assertEqual(value,matrix); self.assertEqual(raw,before)
        self.assertTrue(record['changed'])
        self.assertEqual(record['raw_action_count'],record['normalized_action_count'])
        self.assertEqual(decode_matrix(value,book,d['alignment'],lib),decode_matrix(matrix,book,d['alignment'],lib))

    def test_strict_schema_violation_is_not_hidden_by_type_or_wrapper_adapter(self):
        _,lib,_,_,matrix,_=fixtures.inputs()
        bad=copy.deepcopy(matrix); bad['matrix'][0][0]='1'
        for raw in (bad,matrix['matrix']):
            with self.subTest(raw=raw), self.assertRaises(MatrixTypeError):
                normalize(raw,'matrix',lib,matrix_contract='strict_schema')

    def test_compatibility_never_maps_scene_id_to_object_code(self):
        d,lib,_,book,matrix,_=fixtures.inputs()
        raw=copy.deepcopy(matrix); raw['matrix'][0][2]='999'
        value,_=normalize(raw,'matrix',lib,matrix_contract='typed_json')
        self.assertEqual(value['matrix'][0][2],999)
        with self.assertRaises(PlanError): decode_matrix(value,book,d['alignment'],lib)

    def test_strict_rejection_evidence_is_recorded_before_dispatch(self):
        _,lib,_,_,matrix,_=fixtures.inputs()
        raw=copy.deepcopy(matrix); raw['matrix'][0][0]='1'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(MatrixTypeError):
                normalize_and_record(raw,'matrix',lib,root,'proposal','original.json',matrix_contract='strict_schema')
            note=json.loads((root/'proposal.normalization.json').read_text())
            self.assertEqual(note['status'],'rejected_type_contract')
            self.assertFalse(note['changed'])

    def test_unknown_policy_never_silently_uses_legacy(self):
        _,lib,_,_,matrix,_=fixtures.inputs()
        with self.assertRaises(ValueError):
            normalize(matrix,'matrix',lib,matrix_contract='typo')

    def test_nonfinite_number_rejection_still_records_evidence(self):
        _,lib,_,_,matrix,_=fixtures.inputs()
        raw=copy.deepcopy(matrix); raw['matrix'][0][0]=float('nan')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(MatrixTypeError):
                normalize_and_record(raw,'matrix',lib,root,'proposal','original.json',matrix_contract='strict_schema')
            note=json.loads((root/'proposal.normalization.json').read_text())
            self.assertEqual(note['status'],'rejected_type_contract')
            self.assertIsNotNone(note['raw_sha256'])


if __name__=='__main__': unittest.main()
