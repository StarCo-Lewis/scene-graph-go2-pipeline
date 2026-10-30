"""Width-only offline comparison. No live requests or substitute model answers."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import dialogue_execution_feedback as feedback
import test_typed9_integration as integration
import test_typed9_actions as actions
import test_structural_matrix as structural
from test_typed9_ids import typed_fixture,bound_fixture
from typed_matrix_ids import (code,require,allocation,decode_decomposition,without_context,
    protocol,pool,decomp_prompt,CAPACITY)
from matrix_type_contract import response_format,enforce,MatrixTypeError
from align_objects import call_llm,validate_decomposition
from llm_graph_binding import request_bindings,validate_bindings
from structural_matrix import from_binding_matrix
from action_matrix import make_codebook,decode_matrix,matrix_payload,request_matrix

MODE='typed5_schema'


def action_fixture():
    a9,lib,state,b9,raw9,retrieval,_,_=actions.inputs()
    a5=copy.deepcopy(a9); context=a5['task_decomposition']['identifier_context']
    context['version']=protocol(5)
    context['allocation_sha256']=allocation(context['plan_id'],5)['sha256']
    for key in ('role_codes','task_codes'):
        context[key]={k:code(v//100000000,v%100000000,5) for k,v in context[key].items()}
    b5=make_codebook(a5,lib)
    maps={key:{old['code']:new['code'] for old,new in zip(b9[key],b5[key])}
        for key in ('objects','values','skills','subtasks')}
    slots=dict(maps['objects']); slots.update(maps['values']); slots[0]=0
    raw5=dict(schema_version='action_matrix_typed5_v1',codebook_sha256=b5['sha256'],
        matrix=[[maps['subtasks'][r[0]],maps['skills'][r[1]],slots[r[2]],slots[r[3]]] for r in raw9['matrix']])
    return a5,lib,b5,raw5,retrieval,a9,b9,raw9


class TypedFiveTests(unittest.TestCase):
    def test_five_digit_range_and_disjoint_types(self):
        for kind in range(1,9):
            self.assertEqual(len(str(code(kind,1,5))),5)
            self.assertEqual(require(code(kind,9999,5),kind,digits=5),kind*10000+9999)
        for serial in (0,10000,'1',True):
            with self.subTest(serial=serial),self.assertRaises(ValueError): code(2,serial,5)

    def test_cross_width_numbers_never_coerced(self):
        for value,kind,digits in ((200000001,2,5),(20001,2,9),('20001',2,5),(20001.,2,5)):
            with self.subTest(value=value,digits=digits),self.assertRaises(ValueError): require(value,kind,digits=digits)

    def test_allocation_capacity_is_identical_to_nine_digit(self):
        a5=allocation('paired-fixture',5); a9=allocation('paired-fixture',9)
        self.assertEqual(a5['capacity'],a9['capacity']); self.assertEqual(a5['capacity'],CAPACITY)
        self.assertEqual(len(a5['entries']),len(a9['entries']))
        self.assertNotEqual(a5['sha256'],a9['sha256'])
        self.assertTrue(all(len(str(e['code']))==5 for e in a5['entries']))

    def decomp(self,digits=5):
        book=allocation('paired-decomposition',digits)
        raw=typed_fixture(structural.sample(),book)
        d=validate_decomposition(decode_decomposition(raw,book))
        return d,raw,book

    def test_decoded_decomposition_semantics_are_identical(self):
        d5,raw5,_=self.decomp(5); d9,_,_=self.decomp(9)
        self.assertEqual(raw5['constraints'][0][5],-1)
        self.assertEqual(raw5['constraints'][1][5],0)
        for d in (d5,d9):
            d.pop('identifier_context'); d.pop('structural_matrix_schema')
        self.assertEqual(d5,d9)

    def test_state_subject_still_must_be_robot(self):
        _,raw,book=self.decomp(); raw['constraints'][1][2]=20001
        with self.assertRaisesRegex(ValueError,'KIND_MISMATCH'): decode_decomposition(raw,book)

    def test_strict_mode_does_not_convert_string_codes(self):
        _,raw,book=self.decomp(); raw['roles'][0][0]='20001'
        with self.assertRaises(MatrixTypeError): enforce(raw,'decomposition',MODE,book)

    def test_decomposition_actual_request_issues_five_digit_book(self):
        def request(req,**kw):
            p=json.loads(req.data); data=json.loads(p['messages'][1]['content'])
            book=data['identifier_allocation']
            self.assertEqual(book['schema_version'],'typed5_ids_v1')
            self.assertIn('10001',p['messages'][0]['content'])
            self.assertNotIn('100000001',p['messages'][0]['content'])
            enum=p['response_format']['json_schema']['schema']['properties']['roles']['items']['prefixItems'][0]['enum']
            self.assertEqual(enum,pool(book,2)); self.assertTrue(all(len(str(n))==5 for n in enum))
            raw=typed_fixture(structural.sample(),book)
            return dict(choices=[dict(message=dict(content=json.dumps(raw)),finish_reason='stop')])
        with patch.dict(os.environ,{'OFFLINE_FIVE_KEY':'dummy'}),patch('align_objects.request_json',side_effect=request):
            d=call_llm(structural.sample()['instruction'],'offline','https://invalid.test','OFFLINE_FIVE_KEY',
                max_attempts=1,decomposition_format='matrix',matrix_contract=MODE)
            self.assertEqual(d['identifier_context']['version'],'typed5_ids_v1')

    def test_original_text_with_long_numbers_is_not_globally_rewritten(self):
        book=allocation('text-fixture',5); old=structural.sample()
        old['instruction']='Explicit text identifier 200000001 must remain unchanged.'
        old['roles'][0][2]='Computer mouse with label 200000001'
        raw=typed_fixture(old,book); d=decode_decomposition(raw,book)
        self.assertEqual(d['instruction'],old['instruction'])
        self.assertEqual(d['subtasks'][0]['required_roles'][0]['semantic_query_en'],old['roles'][0][2])

    def test_binding_preserves_original_scene_ids(self):
        for digits in (5,9):
            d,_,_=self.decomp(digits); nodes=structural.inventory()
            raw,book=bound_fixture(d['task_roles'],nodes,d['identifier_context'],
                {'computer mouse':152,'workbench':165,'tape dispenser':230})
            decoded=from_binding_matrix(raw,d['task_roles'],nodes,book)
            bound,errors=validate_bindings(decoded,d['task_roles'],nodes)
            self.assertFalse(errors); self.assertEqual(bound['role_001']['element_ids'],[152])
            self.assertTrue(all(len(str(c))==digits for c in book['active_scene_ids']))

    def test_binding_actual_request_uses_five_digit_schema(self):
        d,_,_=self.decomp(); nodes=structural.inventory()
        raw,book=bound_fixture(d['task_roles'],nodes,d['identifier_context'],
            {'computer mouse':152,'workbench':165,'tape dispenser':230})
        response=dict(choices=[dict(message=dict(content=json.dumps(raw)),finish_reason='stop')])
        with patch.dict(os.environ,{'OFFLINE_FIVE_KEY':'dummy'}),patch('llm_graph_binding.request_json',return_value=response) as req:
            request_bindings(d['task_roles'],nodes,'offline','https://invalid.test','OFFLINE_FIVE_KEY',
                output_format='matrix',matrix_contract=MODE,identifier_context=d['identifier_context'])
            p=json.loads(req.call_args.args[0].data)
            props=p['response_format']['json_schema']['schema']['properties']
            self.assertEqual(props['schema_version']['const'],'object_binding_typed5_v1')
            self.assertEqual(props['matches']['items']['prefixItems'][1]['enum'],[30001,30002,30003])

    def test_decoded_actions_match_nine_digit_exactly(self):
        a5,lib,b5,r5,_,a9,b9,r9=action_fixture()
        self.assertEqual(decode_matrix(r5,b5,a5,lib),decode_matrix(r9,b9,a9,lib))
        self.assertEqual(len(r5['matrix']),len(r9['matrix']))

    def test_action_payload_only_owned_codes_change(self):
        a5,lib,b5,r5,retrieval,a9,b9,r9=action_fixture()
        p5=matrix_payload(a5,lib,b5,'offline',retrieval,False,matrix_contract=MODE)
        p9=matrix_payload(a9,lib,b9,'offline',retrieval,False,matrix_contract='typed9_schema')
        d5=json.loads(p5['messages'][1]['content']); d9=json.loads(p9['messages'][1]['content'])
        for key in ('instruction','goal_state','execution_contract','widened_inventory','zero_rule'):
            self.assertEqual(d5[key],d9[key])
        for a,b in zip(d5['object_codes'],d9['object_codes']):
            self.assertEqual(a['element_id'],b['element_id']); self.assertEqual(a['role_id'],b['role_id'])
            self.assertEqual(len(str(a['code'])),5)
        self.assertEqual(p5['response_format']['json_schema']['schema']['properties']['schema_version']['const'],
                         'action_matrix_typed5_v1')

    def test_mixed_width_mode_and_codebook_rejected(self):
        a5,lib,b5,r5,retrieval,a9,b9,r9=action_fixture()
        for a,b,mode in ((a5,b5,'typed9_schema'),(a9,b9,MODE)):
            with self.assertRaisesRegex(ValueError,'width differs'):
                matrix_payload(a,lib,b,'offline',retrieval,False,matrix_contract=mode)
        with self.assertRaises(ValueError): response_format('decomposition',MODE,allocation('wrong-width',9))

    def test_stale_hash_and_nine_digit_rows_cannot_be_relabelled_five(self):
        a5,lib,b5,r5,_,a9,b9,r9=action_fixture()
        for raw in (dict(r5,codebook_sha256=b9['sha256']),dict(r5,matrix=r9['matrix'])):
            with self.assertRaises(ValueError): decode_matrix(raw,b5,a5,lib)

    def test_typed_five_action_request_uses_same_limits(self):
        a5,lib,b5,r5,retrieval,_,_,_=action_fixture()
        response=dict(choices=[dict(message=dict(content=json.dumps(r5)),finish_reason='stop')])
        with patch.dict(os.environ,{'OFFLINE_FIVE_KEY':'dummy'}),patch('action_matrix.request_json_with_retry',return_value=response) as req:
            request_matrix(a5,lib,b5,'offline','https://invalid.test','OFFLINE_FIVE_KEY',retrieval,
                timeout=180,attempts=2,matrix_contract=MODE)
            self.assertEqual(req.call_args.args[1:3],(180,2))

    def test_proxy_end_state_matches_nine_digit(self):
        five=integration.TypedIntegrationTests().exercise(True,digits=5)
        nine=integration.TypedIntegrationTests().exercise(True,digits=9)
        self.assertEqual(five['resulting_state'],nine['resulting_state'])
        self.assertEqual(five['successful_action_count'],nine['successful_action_count'])

    def test_ids_do_not_repair_missing_precondition(self):
        result=integration.TypedIntegrationTests().exercise(False,digits=5)
        self.assertEqual(result['error'],'GRIPPER_NOT_EMPTY')

    def test_cli_selects_v7_without_changing_case_or_default_v6(self):
        with tempfile.TemporaryDirectory() as temp:
            results=[]
            for mode in (MODE,'typed9_schema'):
                dest=Path(temp)/mode
                run=subprocess.run([sys.executable,str(Path(__file__).with_name('run_matrix_followup.py')),
                    '--preflight-only','--episode-ids','t02','--matrix-contract',mode,'--output-dir',str(dest)],
                    capture_output=True,text=True)
                self.assertEqual(run.returncode,0,run.stderr)
                results.append(json.loads((dest/'dialogue_summary.json').read_text()))
            five,nine=results
            self.assertEqual(five['profile'],'dialogue_proxy_execution_feedback_v7_typed5')
            self.assertEqual(nine['profile'],'dialogue_proxy_execution_feedback_v6_typed9')
            self.assertEqual(five['case_version'],nine['case_version'])
            self.assertEqual(five['planned_round_count'],6)
            self.assertFalse(five['fresh_llm_calls']); self.assertFalse(five['provider_fallback'])


if __name__=='__main__': unittest.main()
