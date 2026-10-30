"""Private offline feasibility witnesses; never imported by the online planner."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matrix_followup_cases as cases
import run_matrix_followup as runner
import dialogue_execution_feedback as feedback
import run_crossroom_task as pilot
from run_contextual_ambiguity import scene_fixture
from realistic_long_cases import episodes as historical_episodes
import test_realistic_long as old_fixtures


class MatrixFollowupTests(unittest.TestCase):
    def args(self,root):
        args=old_fixtures.RealisticLongTests().args(root,True)
        args.group='all'
        return args

    def test_suite_has_seven_exact_reruns_and_four_new_eight_turn_tasks(self):
        suite=cases.episodes()
        self.assertEqual((len(suite),sum(len(e['rounds']) for e in suite)),(11,74))
        old={e['episode_id']:e for e in historical_episodes()}
        for ep in suite:
            self.assertTrue(all('ID ' not in row['instruction'] for row in ep['rounds']))
            if ep['episode_id'] in old:
                value=copy.deepcopy(ep); value.pop('experiment_group')
                self.assertEqual(value,old[ep['episode_id']])
            else:
                self.assertEqual(len(ep['rounds']),8)
        self.assertEqual({e['scene'] for e in cases.new_episodes()},{'scene200','scene202'})
        self.assertEqual(len({(e['scene'],e['user_location']['room_name']) for e in cases.new_episodes()}),4)

    def test_historical_source_contains_fourteen_protocol_rejections_not_all_syntax(self):
        rows=runner.historical_targets()
        self.assertEqual(len(rows),14)
        self.assertEqual({r['episode_id'] for r in rows},set(cases.RERUN_IDS))
        self.assertEqual(sum(r['category']=='action_matrix_coverage_not_JSON_syntax' for r in rows),2)
        self.assertTrue(all((runner.HISTORICAL/r['source_planning_path']).is_dir() for r in rows))

    def test_new_thirty_two_turns_have_feasible_proxy_witnesses(self):
        for ep in cases.new_episodes():
            cfg,nodes,_,_=scene_fixture(ep['scene'])
            state=pilot.initial_state(nodes,cfg)
            for row in ep['rounds']:
                end,plan,_=old_fixtures.reference(ep,row,state,nodes,cfg)
                audit=cases.evaluate(row['round'],state,end,plan,ep['episode_id'])
                self.assertTrue(audit['reference_checks_all_passed'],(ep['episode_id'],row['round'],audit))
                self.assertTrue(audit['dependency_reference_matched'])
                self.assertTrue(audit['strict_id_visit_sequence_matched'],(ep['episode_id'],row['round'],audit))
                end['backend']['memory_version']=row['round']
                state=end

    def test_preflight_has_no_model_calls_no_future_private_input_and_two_groups(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.args(Path(temp)/'pf')
            with patch.object(feedback,'propose_once') as proposer:
                self.assertEqual(runner.run(args),0)
            proposer.assert_not_called()
            batch=pilot.read_json(args.output_dir/'dialogue_summary.json')
            self.assertEqual((batch['planned_round_count'],batch['processed_round_count']),(74,0))
            self.assertEqual(batch['alignment_wire_format'],'matrix')
            self.assertFalse(batch['fresh_llm_calls'])
            self.assertEqual(batch['experiment_groups']['new_eight_turn_task']['planned_round_count'],32)
            self.assertEqual(batch['experiment_groups']['historical_protocol_whole_chain_rerun']['planned_round_count'],42)
            for ep in cases.episodes():
                root=args.output_dir/'episodes'/f"{ep['episode_id']}_r01/round_1/planning"
                text=(root/'execution_contract.json').read_text(encoding='utf-8')
                self.assertNotIn('expected_placement_events',text)
                for future in ep['rounds'][1:]: self.assertNotIn(future['instruction'],text)
                contract=json.loads(text)
                self.assertEqual(contract['conversation_context']['user_location'],ep['user_location'])

    def test_group_selection_is_not_mixed(self):
        with tempfile.TemporaryDirectory() as temp:
            for group,count,rounds in (('new',4,32),('rerun',7,42)):
                args=self.args(Path(temp)/group); args.group=group
                self.assertEqual(runner.run(args),0)
                batch=pilot.read_json(args.output_dir/'dialogue_summary.json')
                self.assertEqual((batch['planned_episode_count'],batch['planned_round_count']),(count,rounds))

    def test_episode_subset_counts_and_cross_group_rejection(self):
        with tempfile.TemporaryDirectory() as temp:
            args=self.args(Path(temp)/'subset'); args.episode_ids=['n01']
            self.assertEqual(runner.run(args),0)
            batch=pilot.read_json(args.output_dir/'dialogue_summary.json')
            self.assertEqual(batch['planned_round_count'],8)
            self.assertEqual(batch['experiment_groups']['new_eight_turn_task']['episode_ids'],['n01'])
            args=self.args(Path(temp)/'invalid'); args.group='new'; args.episode_ids=['t01']
            with self.assertRaisesRegex(ValueError,'outside'): runner.run(args)


if __name__=='__main__':
    unittest.main()
