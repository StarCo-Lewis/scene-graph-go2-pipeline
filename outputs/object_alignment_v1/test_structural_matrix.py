"""Offline protocol checks. No provider calls, model success or physical claims."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from align_objects import call_llm, validate_decomposition, AlignmentError
from structural_matrix import (from_decomposition_matrix, from_binding_matrix,
    make_binding_codebook, matrix_contract_context)
from llm_graph_binding import request_bindings, request_validated_bindings, validate_bindings, BindingError


def sample():
    return dict(schema_version='structural_matrix_v1',
        instruction='Take the mouse to the workbench and release it; leave the tape dispenser on the workbench.',
        task_name='Move mouse',
        roles=[[1,'computer mouse','computer mouse',1,1], [2,'workbench','workbench',1,1],
               [3,'tape dispenser','tape dispenser',1,1]],
        labels=[[1,'mouse'],[1,'computer mouse'],[2,'table'],[3,'tape dispenser']],
        tasks=[[1,'Go near the mouse and pick it up and lift it.'],
               [2,'Carry the mouse to the workbench.'], [3,'Place the mouse on the workbench and release it.']],
        task_roles=[[1,1],[2,1],[2,2],[3,1],[3,2]], dependencies=[[1,2],[2,3]],
        constraints=[[1,2,1,1,2,-1,1],[2,2,0,4,1,0,1],[3,2,3,1,2,-1,1]])


def inventory():
    return [dict(id=i, object_tag=tag, caption=tag, active=True,
                 bbox_center=[0.,1.,0.], bbox_extent=[.1,.1,.1])
            for i,tag in ((152,'computer mouse'),(165,'workbench'),(230,'tape dispenser'))]


def binding(roles, nodes):
    book = make_binding_codebook(roles, nodes)
    # Use the actual supplied book, not an assumption that stage codes equal scene IDs.
    ids = {'computer mouse':152, 'workbench':165, 'tape dispenser':230}
    value = dict(schema_version='object_binding_matrix_v1', codebook_sha256=book['codebook_sha256'],
        decisions=[[r['role_code'],1] for r in book['roles']],
        matches=[[r['role_code'],ids[r['name']]] for r in book['roles']], candidates=[],
        reasons=[[r['role_code'],'Existing category-compatible active node.'] for r in book['roles']])
    return value, book


class StructuralMatrixTests(unittest.TestCase):
    def setUp(self):
        self.raw = sample()
        self.decoded = validate_decomposition(from_decomposition_matrix(self.raw))
        self.roles, self.nodes = self.decoded['task_roles'], inventory()
        self.bound, self.book = binding(self.roles, self.nodes)

    def test_open_vocabulary_and_order_survive_without_mutating_input(self):
        self.assertEqual(self.raw, sample())
        self.assertEqual(self.decoded['subtasks'][1]['description'], self.raw['tasks'][1][1])
        self.assertEqual(self.decoded['subtasks'][1]['dependencies'], ['st_001'])
        self.assertEqual(self.decoded['subtasks'][1]['required_roles'][0]['semantic_labels'], ['mouse','computer mouse'])

    def test_constraint_only_role_is_grounded_not_given_action(self):
        self.assertEqual([r['role_id'] for r in self.decoded['constraint_roles']], ['role_003'])
        self.assertEqual(len(self.decoded['task_roles']), 3)
        self.assertFalse(any(r['role_id']=='role_003' for t in self.decoded['subtasks'] for r in t['required_roles']))
        self.assertFalse(self.decoded['goal_state'][1]['expected_value'])

    def test_initial_held_lifted_are_assertions_not_state_assignments(self):
        self.raw['constraints'] += [[4,1,0,4,1,1,1],[5,1,0,5,1,1,1]]
        dec = validate_decomposition(from_decomposition_matrix(self.raw))
        self.assertEqual([c['relation'] for c in dec['initial_scene_constraints']], ['held','lifted'])
        self.assertNotIn('held_object_id',dec)
        self.assertEqual(len(dec['subtasks']),3)

    def test_bad_scalar_types_are_not_flattened_or_coerced(self):
        for value in ([['mouse']], {'label':'mouse'}, None, True, 1.0):
            with self.subTest(value=value):
                bad=copy.deepcopy(self.raw); bad['labels'][0][1]=value
                with self.assertRaises(ValueError): from_decomposition_matrix(bad)
        for value in ('1', True, 1.0, 0, -1):
            with self.subTest(quantity=value):
                bad=copy.deepcopy(self.raw); bad['roles'][0][3]=value
                with self.assertRaises(ValueError): from_decomposition_matrix(bad)

    def test_invalid_codes_references_and_duplicate_edges_rejected(self):
        mutations = [('roles',0,0,5),('task_roles',0,1,999),('task_roles',0,0,999),
                     ('dependencies',0,0,2),('constraints',0,4,0),('constraints',0,5,1),
                     ('constraints',1,2,1),('constraints',1,5,-1)]
        for key,index,col,value in mutations:
            with self.subTest(key=key,col=col,value=value):
                bad=copy.deepcopy(self.raw); bad[key][index][col]=value
                with self.assertRaises(ValueError): from_decomposition_matrix(bad)
        for key in ('roles','labels','tasks','task_roles','dependencies','constraints'):
            bad=copy.deepcopy(self.raw); bad[key].append(copy.deepcopy(bad[key][0]))
            with self.subTest(duplicate=key), self.assertRaises(ValueError): from_decomposition_matrix(bad)

    def test_unreferenced_roles_missing_labels_and_goals_rejected(self):
        for key in ('constraints','labels','tasks','roles'):
            bad=copy.deepcopy(self.raw); bad[key]=[]
            with self.subTest(key=key), self.assertRaises(ValueError): from_decomposition_matrix(bad)
        bad=copy.deepcopy(self.raw); bad['roles'].append([4,'chair','chair',1,1]); bad['labels'].append([4,'chair'])
        with self.assertRaisesRegex(ValueError,'unreferenced'): from_decomposition_matrix(bad)

    def test_unknown_root_keys_and_legacy_shape_rejected(self):
        bad=copy.deepcopy(self.raw); bad['subtasks']=[]
        with self.assertRaises(ValueError): from_decomposition_matrix(bad)
        with self.assertRaises(ValueError): from_binding_matrix({'bindings':[]},self.roles,self.nodes,self.book)

    def test_binding_roundtrip_and_serialized_book(self):
        decoded=from_binding_matrix(self.bound,self.roles,self.nodes,json.loads(json.dumps(self.book)))
        bound, errors=validate_bindings(decoded,self.roles,self.nodes)
        self.assertEqual(errors,[])
        self.assertEqual(bound['role_001']['element_ids'],[152])
        self.assertEqual(self.bound['matches'][0],[1,152])

    def test_binding_codebook_controls_stage_namespace_not_definition_order(self):
        roles=list(reversed(self.roles))
        value,book=binding(roles,self.nodes)
        self.assertEqual(book['roles'][0]['role_id'],'role_003')
        decoded=from_binding_matrix(value,roles,self.nodes,book)
        bound,errors=validate_bindings(decoded,roles,self.nodes)
        self.assertFalse(errors)
        self.assertEqual(bound['role_003']['element_ids'],[230])

    def test_multi_instance_role_requires_exact_quantity(self):
        roles=copy.deepcopy(self.roles); roles[0]['quantity']=2
        nodes=self.nodes+[dict(self.nodes[0],id=153)]
        value,book=binding(roles,nodes)
        with self.assertRaises(ValueError): from_binding_matrix(value,roles,nodes,book)
        value['matches'].append([1,153])
        decoded=from_binding_matrix(value,roles,nodes,book)
        bound,errors=validate_bindings(decoded,roles,nodes)
        self.assertEqual(bound['role_001']['element_ids'],[152,153])
        self.assertFalse(errors)

    def test_binding_ambiguous_candidates_never_become_bindings(self):
        self.bound['decisions'][0][1]=3
        self.bound['matches']=self.bound['matches'][1:]
        self.bound['candidates']=[[1,152]]
        decoded=from_binding_matrix(self.bound,self.roles,self.nodes,self.book)
        bound, errors=validate_bindings(decoded,self.roles,self.nodes)
        self.assertNotIn('role_001',bound)
        self.assertEqual(errors[0]['reason'],'ambiguous')
        self.assertFalse(errors[0]['candidate_ids_are_bindings'])

    def test_missing_decision_remains_unresolved(self):
        self.bound['decisions'][0][1]=2; self.bound['matches']=self.bound['matches'][1:]
        decoded=from_binding_matrix(self.bound,self.roles,self.nodes,self.book)
        _,errors=validate_bindings(decoded,self.roles,self.nodes)
        self.assertEqual(errors[0]['reason'],'missing')

    def test_binding_types_conflicts_counts_and_unknown_ids_rejected(self):
        for value in ('152',152.,True,None,[152],999,1):
            bad=copy.deepcopy(self.bound); bad['matches'][0][1]=value
            with self.subTest(value=value), self.assertRaises(ValueError):
                from_binding_matrix(bad,self.roles,self.nodes,self.book)
        for key in ('matches','decisions','reasons'):
            bad=copy.deepcopy(self.bound); bad[key].pop()
            with self.subTest(missing=key), self.assertRaises(ValueError):
                from_binding_matrix(bad,self.roles,self.nodes,self.book)
        bad=copy.deepcopy(self.bound); bad['candidates']=[[1,152]]
        with self.assertRaises(ValueError): from_binding_matrix(bad,self.roles,self.nodes,self.book)
        bad=copy.deepcopy(self.bound); bad['matches'][1][1]=152
        with self.assertRaises(ValueError): from_binding_matrix(bad,self.roles,self.nodes,self.book)

    def test_stale_role_node_or_hash_rejected(self):
        for changed in ('roles','nodes','hash'):
            roles,nodes,value=copy.deepcopy((self.roles,self.nodes,self.bound))
            if changed=='roles': roles[0]['quantity']=2
            if changed=='nodes': nodes[0]['caption']='Changed scene metadata'
            if changed=='hash': value['codebook_sha256']='stale'
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                from_binding_matrix(value,roles,nodes,self.book)

    def test_request_local_contract_projection_retains_state_semantics(self):
        contract={'strict_prompt_v2':'Current runtime state. For flat_roles output, old format.\nSKILL JSON:\nold.\nBEFORE RETURNING final state.', 'current_runtime_state':{'held_object_id':152}}
        before=copy.deepcopy(contract); result=matrix_contract_context(contract)
        self.assertEqual(contract,before)
        self.assertNotIn('SKILL JSON',result['strict_prompt_v2'])
        self.assertIn('Current runtime state',result['strict_prompt_v2'])
        self.assertEqual(result['current_runtime_state'],before['current_runtime_state'])

    @patch.dict(os.environ,{'OFFLINE_MATRIX_KEY':'dummy'})
    def test_decomposition_mock_saves_raw_and_decoded_separately(self):
        response={'choices':[{'message':{'content':json.dumps(self.raw)},'finish_reason':'stop'}]}
        with tempfile.TemporaryDirectory() as temp, patch('align_objects.request_json',return_value=response) as request:
            output=call_llm(self.raw['instruction'],'offline','https://invalid.test','OFFLINE_MATRIX_KEY',
                max_attempts=1,decomposition_format='matrix',diagnostics_dir=Path(temp))
            saved=json.loads((Path(temp)/'llm_decomposition_attempt_1.json').read_text())
            self.assertEqual(json.loads(saved['response_content']),self.raw)
            self.assertTrue((Path(temp)/'llm_decomposition_decoded_attempt_1.json').exists())
            self.assertTrue((Path(temp)/'decomposition_codebook_attempt_1.json').exists())
            self.assertEqual(output['decomposition_wire_format'],'matrix')
            payload=json.loads(request.call_args.args[0].data)
            self.assertIn('structural_matrix_v1',payload['messages'][0]['content'])

    @patch.dict(os.environ,{'OFFLINE_MATRIX_KEY':'dummy'})
    def test_invalid_decomposition_is_retained_without_second_proposal(self):
        bad=copy.deepcopy(self.raw); bad['task_roles'][0][1]=999
        response={'choices':[{'message':{'content':json.dumps(bad)}}]}
        with tempfile.TemporaryDirectory() as temp, patch('align_objects.request_json',return_value=response) as request:
            with self.assertRaises(AlignmentError):
                call_llm('Task','offline','https://invalid.test','OFFLINE_MATRIX_KEY',max_attempts=1,
                         decomposition_format='matrix',diagnostics_dir=Path(temp))
            self.assertEqual(request.call_count,1)
            self.assertTrue((Path(temp)/'llm_decomposition_attempt_1.json').exists())
            self.assertFalse((Path(temp)/'llm_decomposition_decoded_attempt_1.json').exists())

    @patch.dict(os.environ,{'OFFLINE_MATRIX_KEY':'dummy'})
    def test_binding_request_prompt_and_raw_output(self):
        response={'choices':[{'message':{'content':json.dumps(self.bound)},'finish_reason':'stop'}]}
        with patch('llm_graph_binding.request_json',return_value=response) as request:
            output=request_bindings(self.roles,self.nodes,'offline','https://invalid.test','OFFLINE_MATRIX_KEY',
                output_format='matrix',inference_policy={'mode':'bounded_inference'})
            payload=json.loads(request.call_args.args[0].data)
            user=json.loads(payload['messages'][1]['content'])
            self.assertEqual(output,self.bound)
            self.assertNotIn('Return JSON {"bindings"',payload['messages'][0]['content'])
            self.assertNotIn('element_ids is an array',payload['messages'][0]['content'])
            self.assertIn('bounded inference policy',payload['messages'][0]['content'])
            self.assertEqual(user['object_binding_codebook'],self.book)
            self.assertNotIn('roles',user)

    def test_binding_loop_saves_raw_canonical_and_no_retry_for_ambiguity(self):
        self.bound['decisions'][0][1]=3; self.bound['matches']=self.bound['matches'][1:]
        self.bound['candidates']=[[1,152]]
        with tempfile.TemporaryDirectory() as temp, patch('llm_graph_binding.request_bindings',return_value=self.bound) as request:
            _,_,errors,attempt=request_validated_bindings(self.roles,self.nodes,'offline','invalid','KEY',
                output_format='matrix',max_validation_attempts=3,diagnostics_dir=Path(temp))
            self.assertEqual((request.call_count,attempt),(1,1))
            self.assertEqual(errors[0]['reason'],'ambiguous')
            self.assertEqual(json.loads((Path(temp)/'llm_binding_matrix_proposal_attempt_1.json').read_text()),self.bound)
            self.assertIn('bindings',json.loads((Path(temp)/'llm_binding_proposal.json').read_text()))

    def test_binding_bad_wire_is_not_normalized_or_substituted(self):
        self.bound['matches'][0][1]='152'
        with tempfile.TemporaryDirectory() as temp, patch('llm_graph_binding.request_bindings',return_value=self.bound) as request:
            with self.assertRaises(BindingError):
                request_validated_bindings(self.roles,self.nodes,'offline','invalid','KEY',
                    output_format='matrix',max_validation_attempts=1,diagnostics_dir=Path(temp))
            self.assertEqual(request.call_count,1)
            self.assertTrue((Path(temp)/'llm_binding_matrix_proposal_attempt_1.json').exists())
            self.assertFalse((Path(temp)/'llm_binding_proposal.json').exists())

    def test_offline_cli_both_matrix_formats(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            for name,value in (('scene_graph.json',self.nodes),('decomp.json',self.raw),('binding.json',self.bound)):
                (root/name).write_text(json.dumps(value),encoding='utf-8')
            result=subprocess.run([sys.executable,str(Path(__file__).with_name('align_objects.py')),
                '--scene-dir',str(root),'--matching-mode','llm_graph','--decomposition-json',str(root/'decomp.json'),
                '--decomposition-format','matrix','--binding-format','matrix',
                '--binding-proposal-json',str(root/'binding.json'),'--output',str(root/'result.json')],
                capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(result.returncode,0,result.stderr)
            output=json.loads((root/'result.json').read_text())
            self.assertEqual(output['status'],'success')
            self.assertTrue(output['alignment']['offline_fixture'])
            self.assertEqual(output['context_bindings']['role_001']['element_ids'],[152])


if __name__=='__main__':
    unittest.main()
