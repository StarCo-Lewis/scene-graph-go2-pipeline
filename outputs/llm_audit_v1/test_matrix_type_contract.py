import copy
import json
from pathlib import Path
import tempfile
import unittest

from matrix_type_contract import (enforce, enforce_and_record, response_format, schema_for,
    validate_owned_schema, MatrixTypeError)


def sample():
    return dict(schema_version='structural_matrix_v1',instruction='Go to the chair.',task_name='Visit chair',
        roles=[[1,'chair','chair',1,1]],labels=[[1,'chair']],tasks=[[1,'Go near the chair.']],
        task_roles=[[1,1]],dependencies=[],constraints=[[1,2,0,3,1,-1,1]])


class MatrixTypeContractTests(unittest.TestCase):
    def test_strict_request_has_integer_and_string_tuple_positions(self):
        fmt=response_format('decomposition','strict_schema')
        self.assertEqual(fmt['type'],'json_schema')
        self.assertIs(fmt['json_schema']['strict'],True)
        schema=fmt['json_schema']['schema']
        self.assertFalse(schema['additionalProperties'])
        row=schema['properties']['roles']['items']
        self.assertEqual([c['type'] for c in row['prefixItems']],['integer','string','string','integer','integer'])
        self.assertEqual((row['minItems'],row['maxItems']),(5,5))
        validate_owned_schema(sample(),schema)

    def test_compatibility_is_explicit_not_mislabeled_schema(self):
        for mode in ('typed_json','legacy_json'):
            self.assertEqual(response_format('decomposition',mode),{'type':'json_object'})
        with self.assertRaises(ValueError): response_format('decomposition','auto')

    def test_strict_never_converts_quoted_numbers(self):
        raw=sample(); raw['roles'][0][0]='1'; before=copy.deepcopy(raw)
        with self.assertRaises(MatrixTypeError) as caught:
            enforce(raw,'decomposition','strict_schema')
        self.assertEqual(raw,before)
        self.assertEqual(caught.exception.record['status'],'rejected')
        self.assertEqual(caught.exception.record['changes'],[])

    def test_compatibility_converts_numeric_columns_and_preserves_numeric_text(self):
        raw=sample()
        raw['roles']=[['1','1','#computer mouse','1','1']]
        raw['labels']=[['1','1']]
        raw['tasks']=[['1','1']]
        raw['task_roles']=[['1','1']]
        raw['constraints']=[['1','2','0','3','1','-1','1']]
        before=copy.deepcopy(raw)
        value,record=enforce(raw,'decomposition','typed_json')
        self.assertEqual(raw,before)
        self.assertEqual(value['roles'],[[1,'1','#computer mouse',1,1]])
        self.assertEqual(value['labels'],[[1,'1']])
        self.assertEqual(value['constraints'],[[1,2,0,3,1,-1,1]])
        self.assertTrue(record['changed']); self.assertFalse(record['raw_schema_conformant'])
        self.assertFalse(record['semantic_validity_checked'])

    def test_noncanonical_integer_strings_are_rejected(self):
        for cell in ('01','+1','-0','1.0','1e0',' 1','1 ','true','１','1\n','1'*33,'#1'):
            raw=sample(); raw['roles'][0][0]=cell
            with self.subTest(cell=cell), self.assertRaises(MatrixTypeError):
                enforce(raw,'decomposition','typed_json')

    def test_float_boolean_null_and_nested_cells_are_not_coerced(self):
        for cell in (1.,True,None,[1],{'code':1},float('nan')):
            raw=sample(); raw['roles'][0][0]=cell
            with self.subTest(cell=repr(cell)), self.assertRaises(MatrixTypeError):
                enforce(raw,'decomposition','typed_json')

    def test_bad_row_shape_is_not_flattened(self):
        raw=sample(); raw['roles'][0].append(1)
        with self.assertRaises(MatrixTypeError): enforce(raw,'decomposition','typed_json')

    def test_integer_text_is_not_inserted_into_text_cells(self):
        raw=sample(); raw['roles'][0][1]=1
        with self.assertRaises(MatrixTypeError): enforce(raw,'decomposition','typed_json')

    def test_partial_changes_are_discarded_on_later_error(self):
        raw=sample(); raw['roles'][0][0]='1'; raw['roles'][0][3]='1.0'
        before=copy.deepcopy(raw)
        with self.assertRaises(MatrixTypeError) as caught: enforce(raw,'decomposition','typed_json')
        record=caught.exception.record
        self.assertEqual(raw,before)
        self.assertIsNone(record['normalized_sha256'])
        self.assertFalse(record['changed']); self.assertEqual(record['changes'],[])
        self.assertEqual(len(record['attempted_changes_discarded']),1)

    def test_wrong_enum_or_missing_goal_not_filled(self):
        for key in ('enum','goal','extra'):
            raw=sample()
            if key=='enum': raw['roles'][0][4]='2'
            if key=='goal': raw['constraints']=[]
            if key=='extra': raw['subtasks']=[]
            with self.subTest(key=key), self.assertRaises(MatrixTypeError):
                enforce(raw,'decomposition','typed_json')

    def test_unchanged_strict_records_raw_conformance_without_semantic_claim(self):
        value,record=enforce(sample(),'decomposition','strict_schema')
        self.assertEqual(value,sample()); self.assertFalse(record['changed'])
        self.assertTrue(record['raw_schema_conformant'])
        self.assertFalse(record['semantic_validity_checked'])

    def test_recorded_compatibility_never_overwrites_original(self):
        raw=sample(); raw['roles'][0][0]='1'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'original.json').write_text(json.dumps(raw),encoding='utf-8')
            _,record=enforce_and_record(raw,'decomposition','typed_json',root,'proposal')
            self.assertEqual(json.loads((root/'original.json').read_text()),raw)
            self.assertEqual(json.loads((root/'proposal.typed.json').read_text())['roles'][0][0],1)
            self.assertTrue((root/'proposal.type_contract.json').exists())
            self.assertNotEqual(record['raw_sha256'],record['normalized_sha256'])

    def test_rejection_record_written_without_typed_proposal(self):
        raw=sample(); raw['roles'][0][0]='1'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            with self.assertRaises(MatrixTypeError):
                enforce_and_record(raw,'decomposition','strict_schema',root,'proposal')
            self.assertEqual(json.loads((root/'proposal.type_contract.json').read_text())['status'],'rejected')
            self.assertFalse((root/'proposal.typed.json').exists())

    def test_binding_schema_uses_current_ids_and_hash_constant(self):
        book=dict(roles=[dict(role_code=1)],active_scene_ids=[152],codebook_sha256='a'*64)
        schema=schema_for('object_binding',book)
        self.assertEqual(schema['properties']['codebook_sha256']['const'],'a'*64)
        raw=dict(schema_version='object_binding_matrix_v1',codebook_sha256='a'*64,
            decisions=[['1','3']],matches=[],candidates=[['1','152']],reasons=[['1','1']])
        value,_=enforce(raw,'object_binding','typed_json',book)
        self.assertEqual(value['decisions'],[[1,3]])
        self.assertEqual(value['candidates'],[[1,152]])
        self.assertEqual(value['reasons'],[[1,'1']])
        self.assertEqual(value['matches'],[])
        for invalid in ('hash','id'):
            bad=copy.deepcopy(raw)
            if invalid=='hash': bad['codebook_sha256']='COPY_SUPPLIED_HASH'
            else: bad['candidates'][0][1]='999'
            with self.subTest(invalid=invalid), self.assertRaises(MatrixTypeError):
                enforce(bad,'object_binding','typed_json',book)

    def test_action_matrix_keeps_every_row_and_unknown_codes_stay_invalid(self):
        book=dict(subtasks=[dict(code=1)],skills=[dict(code=2)],objects=[dict(code=3)],values=[])
        raw={'matrix':[['1','2','3','0'],['1','2','3','0']]}
        value,_=enforce(raw,'skill_matching_matrix','typed_json',book)
        self.assertEqual(value['matrix'],[[1,2,3,0],[1,2,3,0]])
        raw['matrix'][0][2]='152'
        with self.assertRaises(MatrixTypeError): enforce(raw,'skill_matching_matrix','typed_json',book)

    def test_empty_action_matrix_is_not_replaced_with_reference_actions(self):
        value,_=enforce({'matrix':[]},'skill_matching_matrix','strict_schema')
        self.assertEqual(value,{'matrix':[]})

    def test_empty_inventory_does_not_emit_invalid_empty_enum(self):
        schema=schema_for('object_binding',dict(roles=[],active_scene_ids=[],codebook_sha256='b'*64))
        raw=dict(schema_version='object_binding_matrix_v1',codebook_sha256='b'*64,
            decisions=[],matches=[],candidates=[],reasons=[])
        validate_owned_schema(raw,schema)
        self.assertNotIn('"enum": []',json.dumps(schema))


if __name__=='__main__': unittest.main()
