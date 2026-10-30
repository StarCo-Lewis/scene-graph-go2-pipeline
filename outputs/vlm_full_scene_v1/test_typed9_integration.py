"""Full typed-wire fixtures -> unchanged proxy and failure-prefix commits."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import subprocess
import sys

import dialogue_execution_feedback as feedback
import run_crossroom_task as pilot
import test_dialogue_state_recovery as fixtures
from test_full_matrix_wire import wire
from test_typed9_ids import typed_fixture,bound_fixture
from typed_matrix_ids import allocation,decode_decomposition,MODE,without_context
from align_objects import validate_decomposition
from structural_matrix import from_binding_matrix
from llm_graph_binding import validate_bindings
from action_matrix import make_codebook,matrix_payload


def alignment(instruction,nodes,repaired,digits=9):
    issued=allocation('offline-integration-'+str(repaired),digits)
    raw=typed_fixture(wire(instruction,repaired),issued)
    d=validate_decomposition(decode_decomposition(raw,issued))
    roles=d['task_roles']
    raw,book=bound_fixture(roles,nodes,d['identifier_context'],{'book':86,'desk':2,'bottle':121})
    bound,errors=validate_bindings(from_binding_matrix(raw,roles,nodes,book),roles,nodes)
    assert not errors
    return dict(status='success',task_decomposition=d,context_bindings=bound)


class TypedIntegrationTests(unittest.TestCase):
    def exercise(self,repaired,bad_hash=False,digits=9):
        fixture=fixtures.RecoveryTests()
        cfg,nodes,edges,state,instruction,args=fixture.inputs()
        mode=f'typed{digits}_schema'
        args.memory_mode='execution_feedback'; args.alignment_wire_format='matrix'; args.matrix_contract=mode
        args.inference_mode='bounded_inference'
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            feedback.prepare_feedback(root,dict(round=1,instruction=instruction),state,[],cfg,nodes,edges,args)
            def actions(a,lib,book,*pos,**kw):
                old_book=make_codebook(without_context(a),lib)
                raw=fixture.matrix(old_book,repaired)
                mapping={key:{o['code']:n['code'] for o,n in zip(old_book[key],book[key])}
                         for key in ('subtasks','skills','objects','values')}
                slots=dict(mapping['objects']); slots.update(mapping['values']); slots[0]=0
                result=dict(schema_version=f'action_matrix_typed{digits}_v1',
                    codebook_sha256='old-plan' if bad_hash else book['sha256'],
                    matrix=[[mapping['subtasks'][r[0]],mapping['skills'][r[1]],slots[r[2]],slots[r[3]]] for r in raw['matrix']])
                payload=matrix_payload(a,lib,book,'offline',pilot.read_json(root/'skill_retrieval_semantic_1.json'),False,matrix_contract=mode)
                self.assertEqual(payload['response_format']['json_schema']['schema']['properties']['codebook_sha256']['const'],book['sha256'])
                return result
            with patch.object(feedback,'decompose_and_bind',return_value=alignment(instruction,nodes,repaired,digits)) as decomp, \
                    patch.object(feedback,'request_matrix',side_effect=actions) as request:
                summary,plan,bound=feedback.propose_once(args,root,fixtures.Encoder())
            self.assertEqual((decomp.call_count,request.call_count),(1,1))
            self.assertEqual(summary['profile'],f'dialogue_proxy_execution_feedback_v{7 if digits==5 else 6}_typed{digits}')
            profile=pilot.read_json(root/'experiment_profile.json')
            self.assertEqual(profile['identifier_protocol'],f'typed{digits}_ids_v1')
            self.assertFalse(profile['runtime_replanning']); self.assertFalse(profile['provider_fallback'])
            if bad_hash:
                self.assertIsNone(plan); self.assertEqual(summary['status'],'proposal_unavailable')
                self.assertTrue((root/'llm_matrix_proposal_semantic_1_attempt_1.json').exists())
                return None
            self.assertEqual(summary['status'],'proposal_ready_not_semantically_validated')
            before=copy.deepcopy(state)
            result=feedback.execute_proxy(plan,bound,pilot.read_json(root/'execution_skill_library.json'),nodes,state)
            self.assertEqual(state,before)
            return result

    def test_typed_three_stage_plan_dispatches_unchanged_actions(self):
        result=self.exercise(True)
        self.assertEqual(result['status'],'proxy_actions_completed')
        self.assertIsNone(result['resulting_state']['held_object_id'])
        self.assertEqual(result['resulting_state']['backend']['objects']['86']['support_id'],2)
        self.assertEqual(result['resulting_state']['backend']['objects']['121']['support_id'],2)

    def test_missing_precondition_action_still_fails_not_repaired_by_ids(self):
        result=self.exercise(False)
        self.assertEqual(result['status'],'proxy_action_failed')
        self.assertEqual(result['error'],'GRIPPER_NOT_EMPTY')
        self.assertEqual(result['successful_action_count'],2)
        self.assertEqual(result['resulting_state']['held_object_id'],121)
        self.assertGreater(result['skipped_action_count'],0)

    def test_stale_action_hash_is_not_replaced_by_program(self):
        self.exercise(True,bad_hash=True)

    def test_default_matrix_cli_preflight_is_v6_no_provider_calls(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'preflight'
            command=[sys.executable,str(Path(__file__).with_name('run_matrix_followup.py')),
                '--preflight-only','--episode-ids','n04','--output-dir',str(dest)]
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            summary=pilot.read_json(dest/'dialogue_summary.json')
            self.assertEqual(summary['profile'],'dialogue_proxy_execution_feedback_v6_typed9')
            self.assertFalse(summary['fresh_llm_calls'])
            self.assertEqual(summary['episodes'][0]['matrix_contract'],MODE)
            self.assertEqual(summary['episodes'][0]['processed_round_count'],0)

    def test_explicit_v5_cli_remains_available(self):
        with tempfile.TemporaryDirectory() as temp:
            dest=Path(temp)/'preflight'
            command=[sys.executable,str(Path(__file__).with_name('run_matrix_followup.py')),
                '--preflight-only','--episode-ids','n04','--matrix-contract','strict_schema','--output-dir',str(dest)]
            run=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(run.returncode,0,run.stderr)
            summary=pilot.read_json(dest/'dialogue_summary.json')
            self.assertEqual(summary['profile'],'dialogue_proxy_execution_feedback_v5_typed_matrix')


if __name__=='__main__': unittest.main()
