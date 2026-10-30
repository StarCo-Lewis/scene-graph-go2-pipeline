import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import dialogue_execution_feedback as feedback
import run_crossroom_task as pilot
import plan_live_round as worker
import test_dialogue_state_recovery as fixtures
import test_full_matrix_wire as matrices
from action_matrix import matrix_payload
from matrix_type_contract import enforce


class MatrixContractIntegrationTests(unittest.TestCase):
    def inputs(self,mode):
        fixture=fixtures.RecoveryTests()
        cfg,nodes,edges,state,instruction,args=fixture.inputs()
        args.alignment_wire_format='matrix'; args.matrix_contract=mode
        args.memory_mode='execution_feedback'; args.inference_mode='bounded_inference'
        return fixture,cfg,nodes,edges,state,instruction,args

    def exercise(self,mode,quoted=False):
        fixture,cfg,nodes,edges,state,instruction,args=self.inputs(mode)
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            feedback.prepare_feedback(root,dict(round=1,instruction=instruction),state,[],cfg,nodes,edges,args)
            def decompose(options,*a,**kw):
                self.assertEqual(options.matrix_contract,mode)
                return matrices.alignment(instruction,nodes,True)
            def request(alignment,library,book,*a,**kw):
                self.assertEqual(kw['matrix_contract'],mode)
                retrieval=pilot.read_json(root/'skill_retrieval_semantic_1.json')
                payload=matrix_payload(alignment,library,book,'offline',retrieval,False,matrix_contract=mode)
                self.assertEqual(payload['response_format']['type'],'json_schema' if mode=='strict_schema' else 'json_object')
                raw=fixture.matrix(book,True)
                return {'matrix':[[str(c) for c in row] for row in raw['matrix']]} if quoted else raw
            with patch.object(feedback,'decompose_and_bind',side_effect=decompose) as upstream, \
                    patch.object(feedback,'request_matrix',side_effect=request) as actions:
                summary,plan,alignment=feedback.propose_once(args,root,fixtures.Encoder())
            self.assertEqual((upstream.call_count,actions.call_count),(1,1))
            profile=pilot.read_json(root/'experiment_profile.json')
            self.assertIn('v5_typed_matrix',profile['profile'])
            self.assertEqual(profile['matrix_contract'],mode)
            self.assertFalse(profile['provider_fallback'])
            self.assertFalse(profile['runtime_replanning'])
            self.assertEqual(pilot.read_json(root/'runtime_state.json'),state)
            raw=pilot.read_json(root/'llm_matrix_proposal_semantic_1_attempt_1.json')
            if quoted:
                self.assertIs(type(raw['matrix'][0][0]),str)
            if mode=='strict_schema' and quoted:
                self.assertEqual(summary['status'],'proposal_unavailable')
                self.assertIsNone(plan)
                note=pilot.read_json(root/'llm_skill_semantic_1_attempt_1.normalization.json')
                self.assertEqual(note['status'],'rejected_type_contract')
            else:
                self.assertEqual(summary['status'],'proposal_ready_not_semantically_validated')
                self.assertEqual(summary['format_normalization']['changed'],quoted)
                result=feedback.execute_proxy(plan,alignment,pilot.read_json(root/'execution_skill_library.json'),nodes,state)
                self.assertEqual(result['status'],'proxy_actions_completed')

    def test_strict_profile_dispatches_only_conformant_matrix(self):
        self.exercise('strict_schema')

    def test_strict_profile_does_not_hide_provider_type_violation(self):
        self.exercise('strict_schema',True)

    def test_compatibility_profile_audits_quoted_numbers_without_plan_repair(self):
        self.exercise('typed_json',True)

    def test_subprocess_does_not_drop_contract_selection(self):
        _,_,nodes,_,_,instruction,args=self.inputs('strict_schema')
        with tempfile.TemporaryDirectory() as temp:
            args.output_dir=Path(temp); args.alignment_validation_attempts=1
            expected=matrices.alignment(instruction,nodes,True)
            pilot.write_json(args.output_dir/'object_alignment.json',expected)
            def launch(command,**kw):
                self.assertEqual(command[command.index('--matrix-contract')+1],'strict_schema')
                proc=unittest.mock.Mock(); proc.stdout=[]; proc.wait.return_value=0
                return proc
            with patch.object(worker.subprocess,'Popen',side_effect=launch):
                self.assertEqual(worker.decompose_and_bind(args,{'instruction':instruction},{'stages':[]}),expected)


if __name__=='__main__': unittest.main()
