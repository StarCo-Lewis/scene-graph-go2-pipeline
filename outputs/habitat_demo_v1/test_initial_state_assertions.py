"""Initial state assertions are read-only checks, not graph facts or repair actions."""
import copy
import unittest
import json
from pathlib import Path
from unittest.mock import patch

from test_json_long_horizon import nodes, row
from run_json_long_horizon import initial_state
from runtime_constraints import check_runtime_initial_constraints, RuntimeConstraintError
from json_symbolic_review import review_symbolic
from sequential_proxy_core import SerialExecutor
from align_objects import validate_constraints, validate_decomposition, AlignmentError
from decomposition_wire import from_flat_roles
from llm_graph_binding import partition_initial_constraints, check_binding_relations


def predicate(relation='held',value=True,required=True):
    return dict(constraint_id='init_'+relation,subject_role='robot',relation=relation,
                object_role='kettle',expected_value=value,required=required)


class InitialAssertionsTests(unittest.TestCase):
    def fixture(self):
        state=initial_state(nodes())
        state.update(held_object_id=67,lifted=True,prepared_target_id=None)
        alignment={'context_bindings':{'kettle':{'element_ids':[67]}},
                   'task_decomposition':{'initial_scene_constraints':[predicate(),predicate('lifted')]}}
        return alignment,state

    def test_accept_exact_failed_shape_and_check_input_without_mutation(self):
        alignment,state=self.fixture()
        declarations=alignment['task_decomposition']['initial_scene_constraints']
        normalized=validate_constraints(declarations,'initial_scene_constraints',{'kettle'})
        graph,runtime=partition_initial_constraints(normalized)
        self.assertEqual(graph,[])
        self.assertEqual(len(runtime),2)
        # No graph edge required and neither declaration is deleted.
        _,errors=check_binding_relations(alignment['context_bindings'],normalized,[])
        self.assertEqual(errors,[])
        before=copy.deepcopy(state)
        checks=check_runtime_initial_constraints(alignment,state,1.5)
        self.assertEqual([c['status'] for c in checks],['confirmed_in_runtime_snapshot']*2)
        self.assertTrue(all(c['state_assignment_performed'] is False for c in checks))
        self.assertEqual(state,before)
        self.assertEqual(alignment['task_decomposition']['initial_scene_constraints'],declarations)

    def test_false_or_unknown_initial_conditions_do_not_pass_or_change_state(self):
        alignment,state=self.fixture()
        bad_states=[dict(state,held_object_id=None,lifted=False),dict(state,lifted=False),
                    dict(state,held_object_id=13),dict(state,held_object_id=None,lifted=True),
                    dict(state,held_object_id=True),{k:v for k,v in state.items() if k!='held_object_id'},
                    {k:v for k,v in state.items() if k!='lifted'}]
        for bad in bad_states:
            before=copy.deepcopy(bad)
            with self.subTest(bad=bad.get('held_object_id')),self.assertRaises(RuntimeConstraintError):
                check_runtime_initial_constraints(alignment,bad,1.5)
            self.assertEqual(bad,before)

    def test_negative_unknown_and_unbound_are_not_assumed_true(self):
        alignment,state=self.fixture()
        alignment['task_decomposition']['initial_scene_constraints']=[predicate(value=False)]
        with self.assertRaises(RuntimeConstraintError):
            check_runtime_initial_constraints(alignment,state,1.5)
        state.update(held_object_id=None,lifted=False)
        self.assertEqual(check_runtime_initial_constraints(alignment,state,1.5)[0]['actual_value'],False)
        del state['held_object_id']
        with self.assertRaises(RuntimeConstraintError):
            check_runtime_initial_constraints(alignment,state,1.5)
        alignment,state=self.fixture()
        alignment['context_bindings']={}
        with self.assertRaises(RuntimeConstraintError):
            check_runtime_initial_constraints(alignment,state,1.5)

    def test_optional_mismatch_is_recorded_not_promoted_to_evidence(self):
        alignment,state=self.fixture()
        alignment['task_decomposition']['initial_scene_constraints']=[predicate(value=False,required=False)]
        check=check_runtime_initial_constraints(alignment,state,1.5)[0]
        self.assertEqual(check['status'],'not_satisfied')

    def test_invented_relation_object_actor_and_non_boolean_still_rejected(self):
        for changes in ({'relation':'holding'},{'subject_role':'kettle'},
                        {'expected_value':'true'},{'verified':True}):
            with self.subTest(changes=changes),self.assertRaises(AlignmentError):
                validate_constraints([dict(predicate(),**changes)],'initial_scene_constraints',{'kettle'})

    def test_initial_failure_prevents_all_actions_even_if_plan_would_grasp_later(self):
        fixture,library=row([('navigate_to_object',67),('position_arm_for_object',67),
                             ('grasp_object',67),('lift_held_object',67)],held=True)
        fixture['alignment']['task_decomposition']['initial_scene_constraints']=[predicate()]
        state=initial_state(nodes())
        with patch.object(SerialExecutor,'execute') as execute,self.assertRaises(RuntimeConstraintError):
            review_symbolic(fixture,library,nodes(),state,1.5)
        execute.assert_not_called()

    def test_saved_three_failed_returns_against_their_real_input_if_available(self):
        folder=Path(__file__).resolve().parents[1]/'vlm_full_scene_v1/runs/flat_stress_20261014_235353_658107/trials/c03_r01/round_4/planning'
        if not (folder/'runtime_state.json').is_file():
            self.skipTest('Historical user run not included in this checkout')
        state=json.loads((folder/'runtime_state.json').read_text(encoding='utf-8'))
        before=copy.deepcopy(state)
        for attempt in (1,2,3):
            record=json.loads((folder/f'llm_decomposition_attempt_{attempt}.json').read_text(encoding='utf-8'))
            decomposition=validate_decomposition(from_flat_roles(json.loads(record['response_content'])))
            # Explicit-ID unit grounding, NOT a new model binding/result.
            alignment={'context_bindings':{'water_bottle':{'element_ids':[121]}},
                       'task_decomposition':decomposition}
            checks=check_runtime_initial_constraints(alignment,state,1.5)
            self.assertEqual([c['status'] for c in checks],['confirmed_in_runtime_snapshot']*2)
        self.assertEqual(state,before)


if __name__=='__main__': unittest.main()
