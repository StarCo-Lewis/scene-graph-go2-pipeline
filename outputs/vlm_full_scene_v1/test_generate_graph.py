import copy
import io
import json
import tempfile
import unittest
import urllib.error
from types import SimpleNamespace
from unittest.mock import patch
from pathlib import Path
from generate_graph import validate_nodes, validate_decisions, validate_overview, export, sheet, query, main, digest


class GraphTests(unittest.TestCase):
    def setUp(self):
        self.rows=[{'id':1},{'id':2}]
        self.nodes={'nodes':[{'id':i,'status':'recognized','object_tag':'visual object',
                             'caption':'A visible object','evidence':'Visible outline',
                             'uncertainty_reason':None} for i in (1,2)]}
        self.pairs=[{'source':1,'target':2,'possible_relations':['on'],'render_views':[]}]
        self.edges={'decisions':[{'source':1,'target':2,'relation':'on',
                                 'evidence':'Visible apparent support','both_visible_in_both_views':True}]}

    def test_every_node_required(self):
        value=copy.deepcopy(self.nodes)
        value['nodes'].pop()
        with self.assertRaises(ValueError): validate_nodes(value,self.rows)

    def test_duplicate_rejected(self):
        value=copy.deepcopy(self.nodes)
        value['nodes'][1]['id']=1
        with self.assertRaises(ValueError): validate_nodes(value,self.rows)

    def test_bool_id_rejected(self):
        value=copy.deepcopy(self.nodes)
        value['nodes'][0]['id']=True
        with self.assertRaises(ValueError): validate_nodes(value,self.rows)

    def test_unknown_pair_rejected(self):
        value=copy.deepcopy(self.edges)
        value['decisions'][0]['target']=99
        with self.assertRaises(ValueError): validate_decisions(value,self.pairs)

    def test_uncertain_requires_reason(self):
        value=copy.deepcopy(self.nodes)
        value['nodes'][0]['status']='uncertain'
        with self.assertRaises(ValueError): validate_nodes(value,self.rows)

    def fixture(self):
        return {'coordinate_frame':'source_blender_world_z_up_m','instance_count':2,
                'relation_candidates':self.pairs,'inventory':[
                    {'id':i,'geometry_valid':True,'bbox_center':[float(i),0.,.5],
                     'bbox_extent':[.2,.3,.4],'mesh_names':[f'mesh_{i}'],'render_views':[]}
                    for i in (1,2)]}

    def run_export(self, decisions=None, annotations=None):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            summary={}
            complete=export(root,self.fixture(),annotations if annotations is not None else
                            {n['id']:n for n in self.nodes['nodes']},
                            decisions if decisions is not None else self.edges['decisions'],summary)
            folder=root/('scene' if complete else 'provisional_scene')
            return complete,summary,json.loads((folder/'scene_graph.json').read_text()),\
                json.loads((folder/'scene_graph_edges.json').read_text())

    def test_edge_marked_hypothesis(self):
        complete,summary,nodes,edges=self.run_export()
        self.assertTrue(complete)
        self.assertFalse(edges[0]['verified'])
        self.assertEqual(edges[0]['relation'],'a on b')

    def test_missing_pair_does_not_complete(self):
        complete,summary,_,_=self.run_export(decisions=[])
        self.assertFalse(complete)
        self.assertEqual(summary['status'],'partial_not_ready')

    def test_unknown_no_fabricated_edge(self):
        decisions=copy.deepcopy(self.edges['decisions'])
        decisions[0]['relation']='unknown'
        _,_,_,edges=self.run_export(decisions=decisions)
        self.assertEqual(edges,[])

    def test_occluded_relation_rejected(self):
        decisions=copy.deepcopy(self.edges['decisions'])
        decisions[0]['both_visible_in_both_views']=False
        _,_,_,edges=self.run_export(decisions=decisions)
        self.assertEqual(edges,[])

    def test_geometry_gate_rejects_without_coercion(self):
        decisions=copy.deepcopy(self.edges['decisions'])
        decisions[0]['relation']='in'
        _,_,_,edges=self.run_export(decisions=decisions)
        self.assertEqual(edges,[])

    def test_uncertain_preserved_inactive(self):
        annotations={n['id']:copy.deepcopy(n) for n in self.nodes['nodes']}
        annotations[1].update(status='uncertain',uncertainty_reason='Fragment only')
        complete,summary,nodes,edges=self.run_export(annotations=annotations)
        self.assertTrue(complete)
        self.assertFalse(nodes[0]['active'])
        self.assertEqual(summary['uncertain_nodes'],1)
        self.assertEqual(edges,[])

    def test_incomplete_overview_blocks_complete_export(self):
        with tempfile.TemporaryDirectory() as temp:
            manifest=self.fixture()
            manifest['overview_views']=[{'path':'overview.png'}]
            summary={'full_scene_overviews_processed':False}
            self.assertFalse(export(Path(temp),manifest,{n['id']:n for n in self.nodes['nodes']},
                                    self.edges['decisions'],summary))

    def test_raw_invalid_return_preserved_and_resume_cached(self):
        class Response:
            def __init__(self,value): self.value=value
            def __enter__(self): return self
            def __exit__(self,*args): return False
            def read(self): return json.dumps(self.value).encode()
        def reply(content):
            return Response({'choices':[{'message':{'content':json.dumps(content)}}],
                             'usage':{'prompt_tokens':10,'completion_tokens':5}})
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            image=root/'fake.png'
            image.write_bytes(b'fake-test-image-not-rendered')
            args=SimpleNamespace(model='fake-test-model',base_url='https://example.invalid/v1',
                                 attempts=2,max_requests=2,timeout=1)
            budget={'requests':0}
            with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'fake-test-secret'}), \
                 patch('urllib.request.urlopen',side_effect=[reply({'nodes':[]}),reply(self.nodes)]) as network:
                result=query(args,'nodes',self.rows,image,root/'audit',validate_nodes,budget)
                self.assertEqual(result,self.nodes)
                self.assertEqual(network.call_count,2)
                query(args,'nodes',self.rows,image,root/'audit',validate_nodes,budget)
                self.assertEqual(network.call_count,2)
            failed=json.loads((root/'audit/attempt_1.json').read_text())
            self.assertEqual(failed['status'],'validation_rejected')
            self.assertIn('raw_response',failed)
            self.assertNotIn('fake-test-secret',(root/'audit/attempt_1.json').read_text())

    def test_missing_key_no_requests(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            image=root/'fake.png'
            image.write_bytes(b'fake-test')
            args=SimpleNamespace(model='fake',base_url='https://example.invalid',attempts=1,max_requests=1,timeout=1)
            with patch.dict('os.environ',{'DASHSCOPE_API_KEY':''}),patch('urllib.request.urlopen') as network:
                with self.assertRaisesRegex(ValueError,'DASHSCOPE_API_KEY required'):
                    query(args,'nodes',self.rows,image,root/'audit',validate_nodes,{'requests':0})
                network.assert_not_called()

    def test_transport_retry_input_unchanged_and_usage_unknown(self):
        payloads=[]
        def fail(request,timeout):
            payloads.append(request.data)
            raise TimeoutError('fake timeout')
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            image=root/'fake.png'
            image.write_bytes(b'fake-test')
            args=SimpleNamespace(model='fake',base_url='https://example.invalid',attempts=2,max_requests=2,timeout=1)
            with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'fake-secret'}), \
                 patch('urllib.request.urlopen',side_effect=fail):
                with self.assertRaises(ValueError):
                    query(args,'nodes',self.rows,image,root/'audit',validate_nodes,{'requests':0})
            self.assertEqual(payloads[0],payloads[1])
            audit=json.loads((root/'audit/attempt_1.json').read_text())
            self.assertTrue(audit['usage_unknown'])

    def test_all_stages_explicit_json_non_thinking_contract(self):
        for stage,rows,validator in (('nodes',self.rows,validate_nodes),
                                     ('edges',self.pairs,validate_decisions),
                                     ('overview',self.rows,validate_overview)):
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as temp:
                root=Path(temp)
                image=root/'fake.png'
                image.write_bytes(b'fake-test')
                args=SimpleNamespace(model='fake',base_url='https://example.invalid',
                                     attempts=1,max_requests=1,timeout=1)
                with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'fake-secret'}), \
                     patch('urllib.request.urlopen',side_effect=TimeoutError) as network:
                    with self.assertRaises(ValueError):
                        query(args,stage,rows,image,root/'audit',validator,{'requests':0})
                payload=json.loads(network.call_args.args[0].data)
                self.assertFalse(payload['enable_thinking'])
                self.assertFalse(payload['stream'])
                self.assertEqual(payload['response_format'],{'type':'json_object'})
                self.assertIn('json',payload['messages'][0]['content'].lower())
                audit=json.loads((root/'audit/attempt_1.json').read_text())
                self.assertEqual(audit['request_parameters']['enable_thinking'],False)
                self.assertNotIn('messages',audit['request_parameters'])

    def test_http_400_diagnostics_redacted_and_no_retry(self):
        body={'error':{'code':'invalid_parameter','message':
              'JSON mode requires enable_thinking=false; Bearer fake-test-secret; sk-other-secret',
              'api_key':'unrelated-credential'}}
        failure=urllib.error.HTTPError('https://example.invalid',400,'Bad Request',{},
                                      io.BytesIO(json.dumps(body).encode()))
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            image=root/'fake.png'
            image.write_bytes(b'fake-test')
            args=SimpleNamespace(model='fake',base_url='https://example.invalid',attempts=2,max_requests=2,timeout=1)
            with patch.dict('os.environ',{'DASHSCOPE_API_KEY':'fake-test-secret'}), \
                 patch('urllib.request.urlopen',side_effect=failure) as network:
                with self.assertRaisesRegex(RuntimeError,'HTTP 400: JSON mode requires') as caught:
                    query(args,'nodes',self.rows,image,root/'audit',validate_nodes,{'requests':0})
                self.assertEqual(network.call_count,1)
            content=(root/'audit/attempt_1.json').read_text()
            for secret in ('fake-test-secret','sk-other-secret','unrelated-credential'):
                self.assertNotIn(secret,content)
                self.assertNotIn(secret,str(caught.exception))
            audit=json.loads(content)
            self.assertEqual(audit['provider_code'],'invalid_parameter')
            self.assertTrue(audit['usage_unknown'])

    def test_api_smoke_only_one_batch_no_graph_export(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            image=root/'fake.png'
            image.write_bytes(b'fake-test')
            view={'path':'fake.png','sha256':digest(image)}
            manifest=self.fixture()
            manifest.update(status='evidence_complete',source_blend=str(root/'absent.blend'),
                            source_sha256='test',mesh_count=2,limitations=[],overview_views=[view,view])
            for row in manifest['inventory']+manifest['relation_candidates']:
                row['render_views']=[view,view]
            (root/'evidence_manifest.json').write_text(json.dumps(manifest))
            original='original graph summary remains untouched'
            (root/'graph_generation_summary.json').write_text(original)
            def fake_sheet(unused_root,rows,stage,destination):
                destination.parent.mkdir(parents=True,exist_ok=True)
                destination.write_bytes(b'fake-contact-sheet')
            with patch('sys.argv',['generate_graph.py','--sample-dir',str(root),'--api-smoke-only','--attempts','1']), \
                 patch('generate_graph.sheet',side_effect=fake_sheet),patch('generate_graph.query') as request:
                self.assertEqual(main(),0)
                self.assertEqual(request.call_count,1)
            self.assertEqual((root/'graph_generation_summary.json').read_text(),original)
            self.assertFalse((root/'scene').exists())
            smoke=json.loads((root/'api_smoke_summary.json').read_text())
            self.assertEqual(smoke['status'],'api_smoke_passed')
            self.assertFalse(smoke['full_scene_generated'])


if __name__=='__main__': unittest.main()
