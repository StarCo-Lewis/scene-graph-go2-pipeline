"""Prompt isolation tests with offline fixtures only."""
import copy
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import strict_prompt_v2 as v2
import hierarchical_json_planner as hierarchy
import run_hierarchical_json_ab as ab
from run_navigation_demo import read_json, write_json
from run_json_long_horizon import prepare_round, initial_state
from test_json_long_horizon import nodes


class StrictPromptTests(unittest.TestCase):
    def prepare(self, root):
        prepare_round(root, {'round':1,'instruction':'Visit sideboard ID 13.','expected_end':{}},
                      initial_state(nodes()), nodes(), 'offline', 'https://invalid.test')

    def test_contract_changes_text_only_and_is_idempotent(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.prepare(root)
            before={str(p.relative_to(root)):p.read_bytes() for p in root.rglob('*') if p.is_file()}
            policy=read_json(root/'execution_contract.json')
            library=read_json(root/'execution_skill_library.json')
            v2.apply_contract(root)
            new_policy=read_json(root/'execution_contract.json')
            new_library=read_json(root/'execution_skill_library.json')
            self.assertEqual(new_policy.pop('strict_prompt_v2'),v2.CONTRACT)
            self.assertEqual(new_policy,policy)
            self.assertEqual(new_library['skills'],library['skills'])
            self.assertEqual(new_library['execution_contract']['strict_prompt_v2'],v2.CONTRACT)
            for name,content in before.items():
                if name not in ('execution_contract.json','execution_skill_library.json'):
                    self.assertEqual((root/name).read_bytes(),content,name)
            profile=read_json(root/'prompt_profile.json')
            self.assertEqual(profile['sha256'],v2.fingerprints())
            snapshot=(root/'execution_contract.json').read_bytes()
            v2.apply_contract(root)
            self.assertEqual(snapshot,(root/'execution_contract.json').read_bytes())

    def test_v2_hooks_split_and_leaf_only(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            self.prepare(root)
            args=SimpleNamespace(output_dir=root,model='offline',base_url='https://invalid.test')
            limits={'max_depth':3,'max_nodes':23,'max_leaves':12}
            with patch.object(hierarchy,'plan_hierarchical',return_value=0) as plan, \
                 patch.object(hierarchy,'request_decision',return_value={'fixture':True}) as request, \
                 patch.object(hierarchy,'plan_leaf',return_value=0) as leaf:
                self.assertEqual(v2.hierarchical_v2(args,root,limits),0)
                self.assertEqual(plan.call_args.args,(args,root,limits))
                decide=plan.call_args.kwargs['decide']
                check=lambda x:x
                decide(args,root,'hierarchy_split','original system',{'fixture':1},check)
                self.assertEqual(request.call_args.args[3],'original system\n\n'+v2.SPLIT)
                decide(args,root,'hierarchy_coverage','coverage unchanged',{},check)
                self.assertEqual(request.call_args.args[3],'coverage unchanged')
                leaf_fn=plan.call_args.kwargs['leaf_planner']
                leaf_fn(args,reviewer=check)
                self.assertIs(leaf.call_args.kwargs['reviewer'],check)
                self.assertEqual(read_json(root/'execution_contract.json')['strict_prompt_v2'],v2.CONTRACT)

    def test_version_pairing_and_report_are_not_planner_ab(self):
        summary={'status':'completed','experiment':'prompt_v1_v2','fixed_planner':'hierarchical',
            'comparison_arms':['v1','v2'],'positive_trials_per_strategy':1,
            'frozen_engine_unchanged':True,'case_inputs_unchanged':True,
            'trials':[{'case':'test','repeat':1,'kind':'positive','strategy':version,
                       'evaluation_outcome':outcome} for version,outcome in
                      [('v1','final_failure'),('v2','clean_first_pass')]]}
        result=ab.comparison(summary)
        self.assertEqual(set(result['strategies']),{'v1','v2'})
        self.assertEqual(len(result['completed_pairs']),1)
        with tempfile.TemporaryDirectory() as d:
            ab.write_report(Path(d),summary)
            text=(Path(d)/'分层规划对照分析.md').read_text(encoding='utf-8')
            self.assertIn('prompt v1 / v2',text)
            self.assertNotIn('原方案逐原始轮规划',text)
            self.assertNotIn('分层引入更多 LLM 调用',text)

    def test_prompt_is_generic_not_gold_case_answer(self):
        for literal in ('ID 67','ID 13','ID 62','expected_navigation_targets'):
            self.assertNotIn(literal,v2.CONTRACT+v2.SPLIT)
        self.assertIn('Object-object',v2.CONTRACT)
        self.assertIn('explicit visit',v2.CONTRACT)


if __name__=='__main__':
    unittest.main()
