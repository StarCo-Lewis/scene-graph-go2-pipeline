import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import os

import test_action_matrix_ab as fixtures
from action_matrix import make_codebook,decode_matrix,matrix_payload,request_matrix
from matrix_type_contract import enforce,MatrixTypeError
from skill_output_normalization import normalize_and_record
from typed_matrix_ids import allocation,code,MODE,VERSION,without_context


def inputs():
    d,lib,state,old_book,old_raw,retrieval=fixtures.inputs()
    a=d['alignment']; issued=allocation('offline-action-a')
    a['task_decomposition']['identifier_context']=dict(version=VERSION,plan_id=issued['plan_id'],
        allocation_sha256=issued['sha256'],
        role_codes={r:code(2,i) for i,r in enumerate(sorted(a['context_bindings']),1)},
        task_codes={t['subtask_id']:code(5,i) for i,t in enumerate(a['task_decomposition']['subtasks'],1)})
    book=make_codebook(a,lib)
    maps={key:{old['code']:new['code'] for old,new in zip(old_book[key],book[key])}
        for key in ('skills','subtasks','objects','values')}
    args=dict(maps['objects'])
    args.update(maps['values']); args[0]=0
    raw=dict(schema_version='action_matrix_typed9_v1',codebook_sha256=book['sha256'],
        matrix=[[maps['subtasks'][r[0]],maps['skills'][r[1]],args[r[2]],args[r[3]]] for r in old_raw['matrix']])
    return a,lib,state,book,raw,retrieval,old_book,old_raw


class TypedActionTests(unittest.TestCase):
    def test_typed_decode_matches_legacy_actions_exactly(self):
        a,lib,_,book,raw,_,old,old_raw=inputs(); before=copy.deepcopy(raw)
        self.assertEqual(decode_matrix(raw,book,a,lib),decode_matrix(old_raw,old,without_context(a),lib))
        self.assertEqual(raw,before)

    def test_payload_has_no_legacy_codes_in_owned_code_fields(self):
        a,lib,_,book,raw,retrieval,_,_=inputs()
        payload=matrix_payload(a,lib,book,'offline',retrieval,False,matrix_contract=MODE)
        data=json.loads(payload['messages'][1]['content'])
        self.assertTrue(all(t['code']//100000000==5 for t in data['subtasks']))
        self.assertTrue(all(t['code']//100000000==4 for t in data['skills']))
        self.assertTrue(all(t['code']//100000000==8 for t in data['object_codes']))
        self.assertTrue(all(t['code']//100000000==7 for t in data['optional_numeric_preset_codes']))
        for t in data['subtasks']:
            self.assertTrue(all(c//100000000==4 for c in t['candidate_skill_codes']))
        props=payload['response_format']['json_schema']['schema']['properties']
        self.assertEqual(props['codebook_sha256']['const'],book['sha256'])
        self.assertIn('action_matrix_typed9_v1',payload['messages'][0]['content'])
        self.assertNotIn('Only matrix is an output key',payload['messages'][0]['content'])

    def test_wrong_kind_in_each_column_rejected(self):
        a,lib,_,book,raw,_,_,_=inputs()
        for col,value in ((0,code(4,1)),(1,code(5,1)),(2,code(3,1)),(2,code(2,1)),(2,100000001)):
            bad=copy.deepcopy(raw); bad['matrix'][0][col]=value
            with self.subTest(col=col,value=value),self.assertRaises(ValueError): decode_matrix(bad,book,a,lib)

    def test_unissued_code_does_not_pass_just_because_prefix_is_right(self):
        a,lib,_,book,raw,_,_,_=inputs(); raw['matrix'][0][2]=code(8,999)
        with self.assertRaises(ValueError): decode_matrix(raw,book,a,lib)

    def test_raw_scene_id_is_not_automatically_converted(self):
        a,lib,_,book,raw,_,_,_=inputs(); raw['matrix'][0][2]=13
        with self.assertRaises(ValueError): decode_matrix(raw,book,a,lib)

    def test_old_response_and_old_book_are_rejected(self):
        a,lib,_,book,raw,_,_,_=inputs()
        a['task_decomposition']['identifier_context']['plan_id']='offline-action-b'
        fresh=make_codebook(a,lib)
        with self.assertRaises(ValueError): decode_matrix(raw,book,a,lib)
        with self.assertRaises(ValueError): decode_matrix(raw,fresh,a,lib)

    def test_modified_library_invalidates_book(self):
        a,lib,_,book,raw,_,_,_=inputs(); lib['skills'][0]['parameters'][0]['required']=False
        with self.assertRaises(ValueError): decode_matrix(raw,book,a,lib)

    def test_padding_zero_not_robot_or_required_object(self):
        a,lib,_,book,raw,_,_,_=inputs(); raw['matrix'][0][2]=0
        with self.assertRaises(ValueError): decode_matrix(raw,book,a,lib)

    def test_string_float_bool_and_nested_cells_are_rejected(self):
        a,lib,_,book,raw,_,_,_=inputs()
        for value in ('500000001',500000001.,True,[500000001]):
            bad=copy.deepcopy(raw); bad['matrix'][0][0]=value
            with self.subTest(value=value),self.assertRaises(ValueError): decode_matrix(bad,book,a,lib)

    def test_missing_subtask_and_wrong_order_are_not_repaired(self):
        a,lib,_,book,raw,_,_,_=inputs()
        for rows in (raw['matrix'][:1],list(reversed(raw['matrix']))):
            bad=dict(raw,matrix=rows)
            with self.assertRaises(ValueError): decode_matrix(bad,book,a,lib)

    def test_legacy_mode_cannot_silently_dispatch_typed_book(self):
        a,lib,_,book,raw,retrieval,_,_=inputs()
        for mode in ('legacy_json','strict_schema','typed_json'):
            with self.subTest(mode=mode),self.assertRaisesRegex(ValueError,'no downgrade'):
                matrix_payload(a,lib,book,'offline',retrieval,False,matrix_contract=mode)

    def test_normalization_does_not_remove_hash_or_repair_type(self):
        _,lib,_,_,raw,_,_,_=inputs()
        with tempfile.TemporaryDirectory() as temp:
            value,note=normalize_and_record(raw,'matrix',lib,Path(temp),'proposal','raw.json',matrix_contract=MODE)
            self.assertEqual(value,raw); self.assertFalse(note['changed'])
            raw['matrix'][0][0]=str(raw['matrix'][0][0])
            with self.assertRaises(MatrixTypeError):
                normalize_and_record(raw,'matrix',lib,Path(temp),'bad','raw.json',matrix_contract=MODE)

    def test_multiple_bound_instances_get_distinct_codes(self):
        a,lib,_,_,_,_,_,_=inputs()
        role=next(iter(a['context_bindings'])); a['context_bindings'][role]['element_ids']=[13,999]
        book=make_codebook(a,lib)
        entries=[o for o in book['objects'] if o['role_id']==role]
        self.assertEqual(len(entries),2); self.assertNotEqual(entries[0]['code'],entries[1]['code'])
        self.assertEqual(entries[0]['role_code'],entries[1]['role_code'])

    def test_runtime_auxiliary_binding_gets_program_role_not_null(self):
        a,lib,_,_,_,_,_,_=inputs(); before=copy.deepcopy(a)
        a['context_bindings']['runtime_held_object']={'element_ids':[999]}
        book=make_codebook(a,lib)
        added=next(o for o in book['objects'] if o['role_id']=='runtime_held_object')
        self.assertEqual(added['role_code']//100000000,2)
        self.assertNotIn('runtime_held_object',a['task_decomposition']['identifier_context']['role_codes'])

    def test_numeric_preset_cannot_be_used_as_an_object(self):
        a,lib,_,book,raw,_,_,_=inputs()
        if not book['values']: self.fail('Fixture must include a numeric preset')
        raw['matrix'][0][2]=book['values'][0]['code']
        with self.assertRaises(ValueError): decode_matrix(raw,book,a,lib)

    def test_actual_action_request_uses_typed_schema(self):
        a,lib,_,book,raw,retrieval,_,_=inputs()
        response=dict(choices=[dict(message=dict(content=json.dumps(raw)),finish_reason='stop')])
        with patch.dict(os.environ,{'OFFLINE_TYPED_KEY':'dummy'}),patch('action_matrix.request_json_with_retry',return_value=response) as req:
            result=request_matrix(a,lib,book,'offline','https://invalid.test','OFFLINE_TYPED_KEY',retrieval,matrix_contract=MODE)
            self.assertEqual(result,raw)
            payload=json.loads(req.call_args.args[0].data)
            self.assertEqual(payload['response_format']['json_schema']['schema']['properties']['codebook_sha256']['const'],book['sha256'])


if __name__=='__main__': unittest.main()
