"""Offline three-stage matrix -> unchanged proxy dispatch regression."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dialogue_execution_feedback as feedback
import run_crossroom_task as pilot
import run_realistic_long as runner
import plan_live_round as worker
import test_dialogue_state_recovery as fixtures
from align_objects import validate_decomposition
from structural_matrix import from_decomposition_matrix, make_binding_codebook, from_binding_matrix
from llm_graph_binding import validate_bindings


def wire(instruction, repaired):
    # Authored fixture, never imported or used as a model-answer fallback online.
    tasks=['Pick up the black hardcover book.', 'Carry the book to my desk.', 'Place the book on my desk.']
    if repaired:
        tasks.insert(0,'First place the currently held bottle on my desk to free the gripper.')
    task_roles=[]
    for i in range(1,len(tasks)+1):
        codes=([3,2] if i==1 else [1] if i==2 else [1,2]) if repaired else ([1] if i==1 else [1,2])
        task_roles += [[i,code] for code in codes]
    value=dict(schema_version='structural_matrix_v1',instruction=instruction,task_name='Deliver book',
        roles=[[1,'book','black hardcover book',1,1],[2,'desk','user desk in office_B',1,1]],
        labels=[[1,'book'],[2,'desk']],tasks=[[i,t] for i,t in enumerate(tasks,1)],
        task_roles=task_roles,dependencies=[[i,i+1] for i in range(1,len(tasks))],
        constraints=[[1,2,1,1,2,-1,1],[2,2,0,4,1,0,1]])
    if repaired:
        value['roles'].append([3,'bottle','currently held bottle',1,1]); value['labels'].append([3,'bottle'])
        value['constraints'] += [[3,1,0,4,3,1,1],[4,1,0,5,3,1,1],[5,2,3,1,2,-1,1]]
    return value


def alignment(instruction, nodes, repaired):
    decomposition=validate_decomposition(from_decomposition_matrix(wire(instruction,repaired)))
    roles=decomposition['task_roles']; book=make_binding_codebook(roles,nodes)
    ids={'book':86,'desk':2,'bottle':121}
    raw=dict(schema_version='object_binding_matrix_v1',codebook_sha256=book['codebook_sha256'],
        decisions=[[r['role_code'],1] for r in book['roles']],candidates=[],
        matches=[[r['role_code'],ids[r['name']]] for r in book['roles']],
        reasons=[[r['role_code'],'Compatible node from the current scene inventory.'] for r in book['roles']])
    decoded=from_binding_matrix(raw,roles,nodes,book)
    bindings,errors=validate_bindings(decoded,roles,nodes)
    assert not errors
    return dict(status='success',task_decomposition=decomposition,context_bindings=bindings)


class FullMatrixTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.RecoveryTests()
        self.cfg,self.nodes,self.edges,self.state,self.instruction,self.args=self.fixture.inputs()
        self.args.alignment_wire_format='matrix'
        self.args.inference_mode='bounded_inference'
        self.args.memory_mode='execution_feedback'

    def exercise(self, repaired):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            feedback.prepare_feedback(root,dict(round=1,instruction=self.instruction),self.state,
                [],self.cfg,self.nodes,self.edges,self.args)
            def proposal(*args,**kw):
                self.assertEqual(args[0].alignment_wire_format,'matrix')
                self.assertEqual(args[0].alignment_validation_attempts,1)
                return alignment(self.instruction,self.nodes,repaired)
            def actions(a,lib,book,*args,**kw):
                return self.fixture.matrix(book,repaired)
            with patch.object(feedback,'decompose_and_bind',side_effect=proposal) as upstream, \
                    patch.object(feedback,'request_matrix',side_effect=actions) as skills:
                summary,plan,bound=feedback.propose_once(self.args,root,fixtures.Encoder())
            self.assertEqual((upstream.call_count,skills.call_count),(1,1))
            self.assertEqual(summary['status'],'proposal_ready_not_semantically_validated')
            self.assertEqual(summary['decomposition_output_format'],'matrix')
            self.assertEqual(summary['binding_output_format'],'matrix')
            self.assertEqual(summary['skill_output_format'],'matrix')
            self.assertIn('v4_structural_matrix',summary['profile'])
            self.assertEqual(summary['runtime_redecomposition_count'],0)
            self.assertFalse(summary['whole_instruction_review_requested'])
            profile=pilot.read_json(root/'experiment_profile.json')
            self.assertFalse(profile['runtime_replanning'])
            before=copy.deepcopy(self.state)
            result=feedback.execute_proxy(plan,bound,pilot.read_json(root/'execution_skill_library.json'),
                self.nodes,self.state)
            self.assertEqual(self.state,before)
            return result

    def test_three_matrix_stages_dispatch_valid_explicit_fixture(self):
        result=self.exercise(True)
        self.assertEqual(result['status'],'proxy_actions_completed')
        self.assertIsNone(result['resulting_state']['held_object_id'])
        self.assertEqual(result['resulting_state']['backend']['objects']['86']['support_id'],2)
        self.assertEqual(result['resulting_state']['backend']['objects']['121']['support_id'],2)

    def test_matrix_does_not_fix_missing_capacity_action(self):
        result=self.exercise(False)
        self.assertEqual(result['status'],'proxy_action_failed')
        self.assertEqual(result['error'],'GRIPPER_NOT_EMPTY')
        self.assertEqual(result['resulting_state']['held_object_id'],121)
        self.assertEqual(result['successful_action_count'],2)
        self.assertGreater(result['skipped_action_count'],0)

    def test_live_alignment_subprocess_receives_both_matrix_flags(self):
        with tempfile.TemporaryDirectory() as temp:
            self.args.output_dir=Path(temp)
            self.args.alignment_validation_attempts=1
            expected=alignment(self.instruction,self.nodes,True)
            pilot.write_json(Path(temp)/'object_alignment.json',expected)
            def launch(command,**kwargs):
                self.assertEqual(command[command.index('--decomposition-format')+1],'matrix')
                self.assertEqual(command[command.index('--binding-format')+1],'matrix')
                self.assertEqual(command[command.index('--validation-attempts')+1],'1')
                process=unittest.mock.Mock()
                process.stdout=[]; process.wait.return_value=0
                return process
            with patch.object(worker.subprocess,'Popen',side_effect=launch) as process:
                result=worker.decompose_and_bind(self.args,{'instruction':self.instruction},{'stages':[]})
            self.assertEqual(process.call_count,1)
            self.assertEqual(result,expected)

    def test_offline_preflight_profiles_ten_chains_sixty_rounds(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)/'preflight'
            self.args.output_dir=root; self.args.preflight_only=True
            self.args.repeat=1; self.args.episode_ids=None
            self.args.embedding_python=Path(__file__); self.args.embedding_model_dir=Path(temp)
            self.assertEqual(runner.run(self.args),0)
            summary=pilot.read_json(root/'dialogue_summary.json')
            self.assertEqual(summary['planned_round_count'],60)
            self.assertEqual(summary['planned_episode_count'],10)
            self.assertFalse(summary['fresh_llm_calls'])
            self.assertIn('v4_structural_matrix',summary['profile'])
            self.assertEqual(summary['alignment_wire_format'],'matrix')
            self.assertFalse(summary['planning_semantic_gate'])


if __name__=='__main__':
    unittest.main()
