"""Offline harness tests, not paid model results or scale validation."""
import copy
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run_scale_round2 as experiment
import run_crossroom_task as pilot
import run_contextual_ambiguity as contextual
import scale_round2_cases as suite
import scale_round2_state_decision as state_track
from planning_start_contract import conversation_context


class ScaleRound2Tests(unittest.TestCase):
    def args(self,root,**changes):
        return SimpleNamespace(**dict(dict(output_dir=root,preflight_only=True,case_ids=None,repeat=1,
            model='offline',base_url='https://invalid.test',top_k=2,cosine_threshold=.35,cosine_margin=.05),**changes))

    def test_manifest_covers_five_dimensions_four_levels(self):
        rows = suite.cases()
        self.assertEqual(len(rows),20)
        self.assertEqual(len({c[0] for c in rows}),20)
        for axis in 'OLGCS':
            self.assertEqual({c[0] for c in rows if c[0].startswith(axis)},{axis+str(n) for n in range(4)})
            labels = [suite.evaluation_notes(c[0]) for c in rows if c[0].startswith(axis)]
            self.assertEqual({r['target_level'] for r in labels},{0,1,2,3})
            self.assertEqual(len({r['base_task_id'] for r in labels}),1)
        self.assertIsNone(suite.evaluation_notes('S2')['draft_vector']['task_location'])

    def test_declared_history_is_shared_across_constraint_levels(self):
        cfg,nodes,_,_ = contextual.scene_fixture('scene202')
        original = copy.deepcopy(cfg)
        contexts = [conversation_context(suite.fixture_transform('C'+str(i),cfg),nodes) for i in range(4)]
        self.assertTrue(all(c==contexts[0] for c in contexts))
        self.assertEqual(cfg,original)
        self.assertNotIn('dialogue_history',suite.fixture_transform('O1',cfg)['conversation_context'])
        bad = copy.deepcopy(cfg)
        bad['conversation_context']['dialogue_history'] = 'not an array'
        with self.assertRaisesRegex(ValueError,'array'):
            conversation_context(bad,nodes)

    def test_forty_preflight_units_no_online_or_model_calls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'round2'
            with patch.object(pilot,'LocalEncoder') as encoder,patch.object(pilot,'plan_flat') as planner, \
                    patch.object(state_track,'request_json') as request:
                self.assertEqual(experiment.run(self.args(root)),0)
                encoder.assert_not_called()
                planner.assert_not_called()
                request.assert_not_called()
            report = pilot.read_json(root/'scale_round2_summary.json')
            self.assertEqual(report['status'],'preflight_completed')
            self.assertTrue(report['paired_inputs_match'])
            for mode,g in report['groups'].items():
                self.assertEqual(g['planned_planning_trials'],16)
                self.assertEqual(g['planned_state_questions'],4)
                self.assertEqual(g['preflight_rows'],20)
                self.assertEqual(g['symbolically_valid_plans'],0)
                self.assertEqual(g['state_questions_answered'],0)
                for row in g['rows']:
                    folder = root/mode/row['planning_path']
                    public = pilot.read_json(folder/'execution_contract.json')
                    self.assertNotIn('target_level',json.dumps(public))
                    self.assertNotIn('decision_reference',json.dumps(public))
                    self.assertEqual(set(pilot.read_json(folder/'round_input.json')),{'round','instruction'})

    def test_state_question_live_path_never_calls_planner_or_encoder(self):
        answer = {'premise_entails_water':'yes','premise_entails_drinkability':'yes',
                  'decision':'eligible_under_supplied_premise','robot_verified':False,
                  'evidence':[],'reason':'Offline synthetic response, not model evidence'}
        response = {'choices':[{'message':{'content':json.dumps(answer)}}]}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'round2'
            args = self.args(root,preflight_only=False,case_ids=['S0'],embedding_python=Path(__file__),
                             embedding_model_dir=Path(temporary))
            with patch.dict(os.environ,{'DASHSCOPE_API_KEY':'offline-test-not-a-real-key'}), \
                    patch.object(pilot,'model_fingerprint',return_value='synthetic-fingerprint'), \
                    patch.object(pilot,'LocalEncoder') as encoder,patch.object(pilot,'plan_flat') as planner, \
                    patch.object(state_track,'request_json',return_value=response) as request:
                self.assertEqual(experiment.run(args),0)
                encoder.assert_not_called()
                planner.assert_not_called()
                self.assertEqual(request.call_count,2)
            report = pilot.read_json(root/'scale_round2_summary.json')
            for mode,g in report['groups'].items():
                self.assertEqual(g['planned_planning_trials'],0)
                self.assertEqual(g['state_questions_answered'],1)
                self.assertEqual(g['symbolically_valid_plans'],0)
                row = g['rows'][0]
                state = pilot.read_json(root/mode/row['planning_path']/'runtime_state.json')
                self.assertEqual(state['backend']['memory_version'],0)
                self.assertIsNone(state['held_object_id'])

    def test_unknown_state_is_valid_answer_not_forced_yes(self):
        answer = {'premise_entails_water':'unknown','premise_entails_drinkability':'unknown',
                  'decision':'insufficient_information','robot_verified':False,'evidence':[],'reason':'Unknown'}
        self.assertEqual(state_track.validate(answer),answer)
        self.assertEqual(suite.evaluation_notes('S3')['decision_reference'],'insufficient_information')


if __name__ == '__main__':
    unittest.main()
