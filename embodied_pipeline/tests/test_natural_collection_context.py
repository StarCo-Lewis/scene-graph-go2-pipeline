"""Offline public-context integration; no network, WSL or robot access."""
import copy
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from embodied_pipeline.tests.test_task_collection_worker import CollectionWorkerTests, Encoder
from embodied_pipeline.data_collection import context_contract as ctx
from embodied_pipeline.orchestration import task_collection_worker as worker
from embodied_pipeline._legacy import load


class StopBeforeNetwork(BaseException):
    pass


class NaturalContextTests(CollectionWorkerTests):
    def setUp(self):
        super().setUp()
        context = dict(source='scenario_assumption_not_observation',
            user_location=dict(room_name=None, near_scene_element_id=179),
            user_activity='Waiting near the table.',
            dialogue_history=[dict(speaker='user', text='I mean the table mentioned earlier.')])
        spatial = {key: 'Explicitly authored; do not infer missing facts.' for key in ctx.SPATIAL_KEYS}
        spatial.update(user_exact_position=None, user_heading=None,
                       robot_location=dict(status='unknown', same_as_user=None), room_status='unknown')
        self.task.update(collection_public_context_version=ctx.VERSION,
            conversation_context=context, public_spatial_context=spatial,
            reference={'private': 'PRIVATE_REFERENCE_SENTINEL'}, expected_response='PRIVATE_REFERENCE_SENTINEL',
            model_input=dict(instruction=self.task['instruction'], conversation_context=copy.deepcopy(context),
                public_spatial_context=copy.deepcopy(spatial),
                context_messages=[dict(role='user', content='I mean the table mentioned earlier.')]))

    def test_public_projection_and_three_stage_payloads(self):
        with tempfile.TemporaryDirectory() as temp:
            source = self.source(temp); root = Path(temp)/'task'
            worker.claim(root, self.task)
            library, _, _ = worker.prepare_inputs(self.task, source, root)
            contract = library['execution_contract']
            self.assertEqual(contract['conversation_context'], self.task['conversation_context'])
            self.assertEqual(contract['public_spatial_context'], self.task['public_spatial_context'])
            self.assertNotIn('PRIVATE_REFERENCE_SENTINEL', json.dumps(contract))
            public = worker.read(root/'round_input.json')
            self.assertEqual(set(public), {'instruction','conversation_context','public_spatial_context'})
            captured=[]
            def capture(request, **kwargs):
                captured.append(json.loads(request.data))
                raise StopBeforeNetwork()
            align = load('align_objects')
            with patch.object(align, 'request_json', side_effect=capture), self.assertRaises(StopBeforeNetwork):
                align.call_llm(self.task['instruction'], 'offline', 'https://invalid.example',
                    'DASHSCOPE_API_KEY', max_attempts=1, execution_contract=contract,
                    decomposition_format='matrix', matrix_contract='typed5_schema')
            binding = load('llm_graph_binding')
            alignment=self.alignment()
            with patch.object(binding, 'request_json', side_effect=capture), self.assertRaises(StopBeforeNetwork):
                binding.request_bindings([dict(role_id='table',name='table',semantic_query_en='table',
                    semantic_labels=['table'],quantity=1,required=True)], worker.read(source/'scene_graph.json'),
                    'offline','https://invalid.example','DASHSCOPE_API_KEY', instruction=self.task['instruction'],
                    conversation_context=contract['conversation_context'], factual_boundary=contract['factual_boundary'],
                    matrix_contract='typed5_schema', identifier_context=alignment['task_decomposition']['identifier_context'])
            self.assertEqual(len(captured),2)
            for payload in captured:
                data=json.loads(payload['messages'][-1]['content'])
                public_contract=data.get('execution_contract',data)
                self.assertEqual(public_contract['conversation_context'],self.task['conversation_context'])
                self.assertIn('User and robot are different', public_contract['factual_boundary'])
                self.assertNotIn('PRIVATE_REFERENCE_SENTINEL',json.dumps(payload))

    def matrix(self, alignment, library, book, *args, **kwargs):
        payload=worker.skills.matrix_payload(alignment,library,book,'offline',kwargs['retrieval'],False,
                                             matrix_contract='typed5_schema')
        serialized=json.dumps(payload)
        self.assertIn('I mean the table mentioned earlier.', serialized)
        self.assertNotIn('PRIVATE_REFERENCE_SENTINEL',serialized)
        return super().matrix(alignment,library,book,*args,**kwargs)

    def test_invalid_context_rejected_before_requests(self):
        mutations=[lambda t:t.pop('collection_public_context_version'),
                   lambda t:t['model_input']['conversation_context']['user_location'].update(near_scene_element_id=999),
                   lambda t:t['model_input']['conversation_context'].update(reference='PRIVATE_REFERENCE_SENTINEL'),
                   lambda t:t['model_input'].update(context_messages=[]),
                   lambda t:t['model_input']['public_spatial_context'].update(user_heading=0)]
        for change in mutations:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                task=copy.deepcopy(self.task);change(task)
                with patch.object(worker,'collect_alignment') as alignment:
                    result=worker.run_task(self.args,task,self.source(temp),Path(temp)/'task',Encoder())
                alignment.assert_not_called()
                self.assertFalse(result['engineering_json_generated'])

    def test_d3_unknown_room_accepted_fabricated_room_rejected(self):
        task=copy.deepcopy(self.task);task['scene_id']='d3_619'
        ctx.public_context(task,[{'id':179}])
        for loc in (task['conversation_context']['user_location'],task['model_input']['conversation_context']['user_location']):
            loc['room_name']='invented_room'
        with self.assertRaises(ValueError):ctx.public_context(task,[{'id':179}])

    def test_summary_preserves_unrun_and_failed_records(self):
        from embodied_pipeline.data_collection import natural_batch as natural
        from embodied_pipeline.data_collection import batch
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);scenes={}
            for scene in natural.SCENES:
                task=copy.deepcopy(self.task);task.update(scene_id=scene,record_id=scene+'_10000001',horizon='short')
                path=root/'cases'/scene/'tasks.jsonl';batch.jsonl(path,[task])
                scenes[scene]={'tasks_sha256':batch.sha(path)}
            batch.write(root/'collection_config.json',dict(scenes=scenes,clarification_policy='ungraded'))
            batch.write(root/'cases/scene200/tasks/scene200_10000001/model_run/planning_summary.json',
                        dict(status='collection_stopped_at_stage_ungraded',error='offline timeout'))
            natural.summarize(root)
            summary=batch.read(root/'collection_summary.json')
            self.assertEqual(summary['records'],3)
            self.assertEqual(summary['scenes']['scene202'],{'not_started':1})
            self.assertEqual(summary['engineering_json_count'],0)
