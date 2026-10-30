"""Authored offline fixtures, not model answers or online fallbacks."""
import copy
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from align_objects import call_llm,validate_decomposition,AlignmentError
from structural_matrix import make_binding_codebook,from_binding_matrix,matrix_contract_context
from llm_graph_binding import request_bindings,request_validated_bindings,validate_bindings
from matrix_type_contract import enforce,response_format,MatrixTypeError
from typed_matrix_ids import (allocation,code,require,lookup,pool,decode_decomposition,
    decomp_prompt,ROBOT,CAPACITY,MODE,width)
import test_structural_matrix as fixtures


def typed_fixture(raw,book):
    digits=width(book)
    value=copy.deepcopy(raw); value['schema_version']=f'structural_typed{digits}_v1'
    value['codebook_sha256']=book['sha256']
    for key,kind in (('roles',2),('tasks',5),('constraints',6)):
        for row in value[key]: row[0]=pool(book,kind)[row[0]-1]
    def r(n): return code(1,1,digits) if n==0 else pool(book,2)[n-1]
    for row in value['labels']: row[0]=r(row[0])
    for row in value['task_roles']: row[:]=[pool(book,5)[row[0]-1],r(row[1])]
    for row in value['dependencies']: row[:]=[pool(book,5)[n-1] for n in row]
    for row in value['constraints']: row[2]=r(row[2]); row[4]=r(row[4])
    return value


def bound_fixture(roles,nodes,context,ids):
    book=make_binding_codebook(roles,nodes,context)
    codes={e['element_id']:e['code'] for e in book['scene_codes']}
    raw=dict(schema_version=f'object_binding_typed{width(context)}_v1',codebook_sha256=book['codebook_sha256'],
        decisions=[[r['role_code'],1] for r in book['roles']],candidates=[],
        matches=[[r['role_code'],codes[ids[r['name']]]] for r in book['roles']],
        reasons=[[r['role_code'],'Existing category-compatible scene node.'] for r in book['roles']])
    return raw,book


class TypedIdsTests(unittest.TestCase):
    def setUp(self):
        self.book=allocation('offline-plan-a')
        self.raw=typed_fixture(fixtures.sample(),self.book)

    def test_decimal_types_are_disjoint_and_nine_digits(self):
        self.assertEqual(len({code(k,1) for k in range(1,9)}),8)
        for k in range(1,9):
            self.assertEqual(len(str(code(k,1))),9)
            self.assertEqual(require(code(k,1),k),code(k,1))

    def test_invalid_integer_serial_and_kind_never_coerced(self):
        for k,n in ((0,1),(9,1),(1,0),(1,100000000),(True,1),(1,False),(1,'1')):
            with self.subTest(k=k,n=n),self.assertRaises(ValueError): code(k,n)
        for value in ('100000001',100000001.,True,0,-100000001,100000000):
            with self.subTest(value=value),self.assertRaises(ValueError): require(value,1)

    def test_allocated_lookup_contains_no_scene_or_skill_guesses(self):
        entries=lookup(self.book)
        self.assertEqual(entries[ROBOT]['canonical'],'robot')
        self.assertEqual(len(pool(self.book,2)),CAPACITY['role'])
        self.assertFalse(any(e['kind'] in ('scene_node','skill') for e in entries.values()))

    def test_modified_allocation_is_rejected(self):
        self.book['entries'][1]['code']=200009999
        with self.assertRaisesRegex(ValueError,'MODIFIED'): lookup(self.book)

    def test_decomposition_preserves_text_goals_and_protection(self):
        before=copy.deepcopy(self.raw)
        value=validate_decomposition(decode_decomposition(self.raw,self.book))
        self.assertEqual(self.raw,before)
        self.assertEqual(value['task_roles'][0]['name'],'computer mouse')
        self.assertEqual(value['goal_state'][1]['subject_role'],'robot')
        self.assertFalse(value['goal_state'][1]['expected_value'])
        self.assertEqual(value['constraint_roles'][0]['name'],'tape dispenser')
        self.assertEqual(value['identifier_context']['role_codes']['role_001'],200000001)

    def test_object_role_cannot_be_held_subject(self):
        self.raw['constraints'][1][2]=self.raw['roles'][0][0]
        with self.assertRaisesRegex(ValueError,'KIND_MISMATCH'): decode_decomposition(self.raw,self.book)

    def test_executor_cannot_be_role_definition(self):
        self.raw['roles'][0][0]=ROBOT
        with self.assertRaises(ValueError): decode_decomposition(self.raw,self.book)

    def test_reserved_executor_name_cannot_hide_in_type2_role(self):
        self.raw['roles'][0][1]='robot'
        with self.assertRaisesRegex(ValueError,'Reserved executor'): decode_decomposition(self.raw,self.book)

    def test_unissued_and_undefined_roles_are_not_inferred(self):
        for n in (code(2,999),code(2,20)):
            bad=copy.deepcopy(self.raw); bad['task_roles'][0][1]=n
            with self.subTest(n=n),self.assertRaises(ValueError): decode_decomposition(bad,self.book)

    def test_definitions_cannot_skip_slots(self):
        self.raw['roles'][0][0]=code(2,10)
        with self.assertRaisesRegex(ValueError,'slots in order'): decode_decomposition(self.raw,self.book)

    def test_old_plan_response_rejected_even_when_numeric_codes_overlap(self):
        with self.assertRaises(ValueError): decode_decomposition(self.raw,allocation('offline-plan-b'))

    def test_state_expected_and_spatial_subject_rules_survive(self):
        for idx,col,value in ((1,5,-1),(0,2,ROBOT)):
            bad=copy.deepcopy(self.raw); bad['constraints'][idx][col]=value
            with self.subTest(idx=idx),self.assertRaises(ValueError): decode_decomposition(bad,self.book)

    def test_no_required_final_goal_still_rejected(self):
        for row in self.raw['constraints']: row[1]=1
        with self.assertRaisesRegex(ValueError,'final goal'): decode_decomposition(self.raw,self.book)

    def test_typed_mode_never_converts_string_codes(self):
        self.raw['roles'][0][0]=str(self.raw['roles'][0][0])
        with self.assertRaises(MatrixTypeError): enforce(self.raw,'decomposition',MODE,self.book)

    def test_schema_enumerates_program_ids_and_hash(self):
        fmt=response_format('decomposition',MODE,self.book)
        props=fmt['json_schema']['schema']['properties']
        self.assertTrue(fmt['json_schema']['strict'])
        self.assertEqual(props['codebook_sha256']['const'],self.book['sha256'])
        self.assertEqual(props['roles']['items']['prefixItems'][0]['enum'],pool(self.book,2))

    def test_decomposition_api_uses_saved_program_book_before_output(self):
        def request(req,**kw):
            payload=json.loads(req.data); data=json.loads(payload['messages'][1]['content'])
            book=data['identifier_allocation']
            self.assertTrue((folder/'identifier_allocation.json').exists())
            self.assertEqual(payload['response_format']['json_schema']['schema']['properties']['codebook_sha256']['const'],book['sha256'])
            raw=typed_fixture(fixtures.sample(),book)
            return dict(choices=[dict(message=dict(content=json.dumps(raw)),finish_reason='stop')])
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'OFFLINE_TYPED_KEY':'dummy'}):
            folder=Path(temp)
            with patch('align_objects.request_json',side_effect=request):
                result=call_llm(fixtures.sample()['instruction'],'offline','https://invalid.test','OFFLINE_TYPED_KEY',
                    max_attempts=1,decomposition_format='matrix',matrix_contract=MODE,diagnostics_dir=folder)
            self.assertEqual(result['identifier_context']['version'],'typed9_ids_v1')
            self.assertTrue((folder/'llm_decomposition_attempt_1.json').exists())

    def test_wrong_subject_raw_response_is_retained_without_repair(self):
        def request(req,**kw):
            book=json.loads(json.loads(req.data)['messages'][1]['content'])['identifier_allocation']
            raw=typed_fixture(fixtures.sample(),book); raw['constraints'][1][2]=code(2,1)
            return dict(choices=[dict(message=dict(content=json.dumps(raw)),finish_reason='stop')])
        with tempfile.TemporaryDirectory() as temp,patch.dict(os.environ,{'OFFLINE_TYPED_KEY':'dummy'}),patch('align_objects.request_json',side_effect=request):
            with self.assertRaisesRegex(AlignmentError,'KIND_MISMATCH'):
                call_llm(fixtures.sample()['instruction'],'offline','https://invalid.test','OFFLINE_TYPED_KEY',
                    max_attempts=1,decomposition_format='matrix',matrix_contract=MODE,diagnostics_dir=Path(temp))
            saved=json.loads((Path(temp)/'llm_decomposition_attempt_1.json').read_text())
            self.assertEqual(json.loads(saved['response_content'])['constraints'][1][2],code(2,1))

    def bindings(self):
        d=validate_decomposition(decode_decomposition(self.raw,self.book)); nodes=fixtures.inventory()
        roles=d['task_roles']; raw,book=bound_fixture(roles,nodes,d['identifier_context'],
            {'computer mouse':152,'workbench':165,'tape dispenser':230})
        return d,roles,nodes,raw,book

    def test_typed_binding_decodes_to_original_scene_ids(self):
        d,roles,nodes,raw,book=self.bindings()
        decoded=from_binding_matrix(raw,roles,nodes,book)
        bound,errors=validate_bindings(decoded,roles,nodes)
        self.assertFalse(errors)
        self.assertEqual(bound['role_001']['element_ids'],[152])
        self.assertEqual(book['scene_codes'][0],dict(code=300000001,element_id=152))

    def test_binding_rejects_raw_ids_wrong_types_and_old_hash(self):
        _,roles,nodes,raw,book=self.bindings()
        for key,value in (('id',152),('id',code(2,1)),('hash','old-plan')):
            bad=copy.deepcopy(raw)
            if key=='id': bad['matches'][0][1]=value
            else: bad['codebook_sha256']=value
            with self.subTest(value=value),self.assertRaises(ValueError): from_binding_matrix(bad,roles,nodes,book)

    def test_binding_ambiguity_is_not_forced_to_match(self):
        _,roles,nodes,raw,book=self.bindings()
        raw['decisions'][0][1]=3; raw['candidates']=[raw['matches'].pop(0)]
        decoded=from_binding_matrix(raw,roles,nodes,book)
        bound,errors=validate_bindings(decoded,roles,nodes)
        self.assertNotIn('role_001',bound); self.assertEqual(errors[0]['reason'],'ambiguous')

    def test_binding_actual_api_payload_uses_typed_book(self):
        d,roles,nodes,raw,book=self.bindings()
        response=dict(choices=[dict(message=dict(content=json.dumps(raw)),finish_reason='stop')])
        with patch.dict(os.environ,{'OFFLINE_TYPED_KEY':'dummy'}),patch('llm_graph_binding.request_json',return_value=response) as req:
            actual=request_bindings(roles,nodes,'offline','https://invalid.test','OFFLINE_TYPED_KEY',
                output_format='matrix',matrix_contract=MODE,identifier_context=d['identifier_context'])
            self.assertEqual(actual,raw)
            payload=json.loads(req.call_args.args[0].data)
            props=payload['response_format']['json_schema']['schema']['properties']
            self.assertEqual(props['matches']['items']['prefixItems'][1]['enum'],book['active_scene_ids'])

    def test_retry_keeps_typed_schema_and_same_scope(self):
        seen=[]
        def request(req,**kw):
            p=json.loads(req.data); data=json.loads(p['messages'][1]['content'].split('\nPrevious output')[0])
            book=data['identifier_allocation']; seen.append(book['sha256'])
            raw=typed_fixture(fixtures.sample(),book)
            if len(seen)==1: raw['constraints'][1][2]=code(2,1)
            else: self.assertIn('Return complete structural_typed9_v1',p['messages'][1]['content'])
            return dict(choices=[dict(message=dict(content=json.dumps(raw)),finish_reason='stop')])
        with patch.dict(os.environ,{'OFFLINE_TYPED_KEY':'dummy'}),patch('align_objects.request_json',side_effect=request):
            call_llm(fixtures.sample()['instruction'],'offline','https://invalid.test','OFFLINE_TYPED_KEY',
                max_attempts=2,decomposition_format='matrix',matrix_contract=MODE)
        self.assertEqual(len(seen),2); self.assertEqual(seen[0],seen[1])

    def offline_cli(self,include_allocation):
        _,_,nodes,raw,_=self.bindings()
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name,value in (('scene_graph.json',nodes),('decomp.json',self.raw),
                    ('binding.json',raw),('allocation.json',self.book)):
                (root/name).write_text(json.dumps(value),encoding='utf-8')
            command=[sys.executable,str(Path(__file__).with_name('align_objects.py')),
                '--scene-dir',str(root),'--matching-mode','llm_graph','--decomposition-json',str(root/'decomp.json'),
                '--decomposition-format','matrix','--binding-format','matrix','--matrix-contract',MODE,
                '--binding-proposal-json',str(root/'binding.json'),'--output',str(root/'result.json')]
            if include_allocation: command+=['--identifier-allocation-json',str(root/'allocation.json')]
            run=subprocess.run(command,capture_output=True,text=True,encoding='utf-8')
            result=json.loads((root/'result.json').read_text())
            return run,result

    def test_offline_cli_with_saved_allocation_preserves_scene_ids(self):
        run,result=self.offline_cli(True)
        self.assertEqual(run.returncode,0,run.stderr)
        self.assertEqual(result['status'],'success')
        self.assertTrue(result['alignment']['offline_fixture'])
        self.assertEqual(result['context_bindings']['role_001']['element_ids'],[152])

    def test_offline_cli_never_infers_missing_saved_allocation(self):
        run,result=self.offline_cli(False)
        self.assertEqual(run.returncode,1)
        self.assertIn('--identifier-allocation-json',result['error']['message'])

    def test_capacity_overflow_is_rejected_not_truncated(self):
        before=len(self.raw['roles'])
        self.raw['roles']*=CAPACITY['role']
        with self.assertRaises(ValueError): decode_decomposition(self.raw,self.book)
        self.assertEqual(len(self.raw['roles']),before*CAPACITY['role'])

    def test_contract_projection_does_not_request_legacy_wire(self):
        raw={'strict_prompt_v2':'State. For flat_roles output, old. BEFORE RETURNING final checks.'}
        value=matrix_contract_context(raw,'structural_typed9_v1')
        self.assertIn('structural_typed9_v1',value['strict_prompt_v2'])
        self.assertNotIn('structural_matrix_v1',value['strict_prompt_v2'])
        self.assertIn('final checks',value['strict_prompt_v2'])


if __name__=='__main__': unittest.main()
