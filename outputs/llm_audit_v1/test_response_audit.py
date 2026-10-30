import io
import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE)); sys.path.insert(0,str(HERE.parent/'object_alignment_v1'))
sys.path.insert(0,str(HERE.parent/'skill_matching_v1'))
from response_audit import ResponseAudit, validation, category
from build_report import build
from llm_transport import request_json
from match_skills import request_json_with_retry, request_plan, PlanError


def request(secret='test-secret-token'):
    return urllib.request.Request('https://invalid.test/chat/completions',
        data=json.dumps({'model':'test','messages':[{'role':'user','content':'test input'}]}).encode(),
        headers={'Authorization':'Bearer '+secret,'Content-Type':'application/json'})


class ResponseTests(unittest.TestCase):
    def records(self,root):
        return [json.loads(p.read_text()) for p in (Path(root)/'llm_response_audit').glob('*.json')]
    def test_raw_malformed_json_saved_before_parse(self):
        with tempfile.TemporaryDirectory() as root:
            audit=ResponseAudit(root,'decomposition',request=request())
            with patch('llm_transport.urllib.request.urlopen',return_value=io.StringIO('not a JSON document')):
                with self.assertRaises(json.JSONDecodeError): request_json(request(),audit=audit)
            raw=next(r for r in self.records(root) if r.get('event')=='http_response')
            self.assertEqual(raw['response_body'],'not a JSON document')
            self.assertTrue(raw['response_received'])
    def test_timeout_then_success_retains_both_attempts(self):
        with tempfile.TemporaryDirectory() as root:
            req=request(); audit=ResponseAudit(root,'object_binding',request=req)
            with patch('llm_transport.urllib.request.urlopen',side_effect=[TimeoutError('timed out'),io.StringIO('{"ok":true}')]) as open_url, patch('llm_transport.time.sleep'):
                self.assertTrue(request_json(req,attempts=2,audit=audit)['ok'])
                self.assertIs(open_url.call_args_list[0].args[0],open_url.call_args_list[1].args[0])
            r=self.records(root)
            error=next(x for x in r if x.get('event')=='transport_failure')
            self.assertFalse(error['response_received']); self.assertNotIn('response_content',error)
            report=build(root,Path(root)/'report')
            self.assertEqual(report['http_attempt_count'],2); self.assertEqual(report['logical_request_count'],1)
            self.assertEqual(report['transport_failure_count'],1)
    def test_nonretry_http_body_retained_and_exception_body_readable(self):
        with tempfile.TemporaryDirectory() as root:
            req=request(); audit=ResponseAudit(root,'binding',request=req)
            denied=urllib.error.HTTPError('https://invalid.test',403,'denied',{},io.BytesIO(b'{"error":"denied"}'))
            with patch('llm_transport.urllib.request.urlopen',side_effect=denied):
                with self.assertRaises(urllib.error.HTTPError) as e: request_json(req,attempts=2,audit=audit)
            self.assertEqual(e.exception.read(),b'{"error":"denied"}')
            report=build(root,Path(root)/'report')
            self.assertEqual(report['http_error_count'],1); self.assertEqual(report['http_attempt_count'],1)
    def test_skill_transport_raw_saved(self):
        with tempfile.TemporaryDirectory() as root:
            req=request(); audit=ResponseAudit(root,'skill_matching',request=req)
            with patch('match_skills.urllib.request.urlopen',return_value=io.StringIO('{"choices":[]}')):
                self.assertEqual(request_json_with_retry(req,audit=audit),{'choices':[]})
            self.assertTrue(any(r.get('response_body')=='{"choices":[]}' for r in self.records(root)))
    def test_credentials_redacted_request_headers_absent(self):
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ,{'DASHSCOPE_API_KEY':'test-secret-token'}):
            audit=ResponseAudit(root,'skill_matching',request=request())
            audit.received(1,'{"message":"Bearer test-secret-token"}')
            validation(root,'skill_matching',1,'rejected','test-secret-token appeared')
            combined=''.join(p.read_text() for p in (Path(root)/'llm_response_audit').glob('*.json'))
            self.assertNotIn('test-secret-token',combined); self.assertNotIn('Authorization',combined)
            self.assertIn('REDACTED_CREDENTIAL',combined)
    def test_unique_request_and_rejection_records_not_overwritten(self):
        with tempfile.TemporaryDirectory() as root:
            for i in range(2):
                audit=ResponseAudit(root,'skill_matching',{'validation_attempt':i+1},request())
                audit.received(1,f'{{"attempt":{i+1}}}')
                validation(root,'skill_matching',i+1,'rejected','unknown skill',evidence='proposal.json')
            report=build(root,Path(root)/'report')
            self.assertEqual(report['logical_request_count'],2); self.assertEqual(report['program_rejection_count'],2)
            self.assertEqual(report['program_rejections_by_category'],{'unknown_skill':2})
    def test_disabled_no_artifacts(self):
        audit=ResponseAudit(None,'decomposition',request=request())
        audit.received(1,'bad json'); audit.transport_error(2,TimeoutError()); audit.parsed('bad','rejected',ValueError())
        self.assertEqual(audit.files,[])
    def test_pending_killed_worker_request_not_invented_response(self):
        with tempfile.TemporaryDirectory() as root:
            ResponseAudit(root,'skill_matching',request=request())
            report=build(root,Path(root)/'report')
            self.assertEqual(report['requests_without_transport_result_count'],1)
            self.assertEqual(report['http_attempt_count'],0)
    def test_invalid_skill_model_content_retained_even_no_proposal_json(self):
        alignment={'status':'success','context_bindings':{'kettle':{'element_ids':[67]}},
                   'task_decomposition':{'instruction':'Pick up kettle','subtasks':[{'subtask_id':'s1','description':'pick up'}]}}
        library={'skills':[]}
        reply=json.dumps({'choices':[{'message':{'content':'not valid JSON'}}]})
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ,{'DASHSCOPE_API_KEY':'test-secret-token'}), \
             patch('match_skills.urllib.request.urlopen',return_value=io.StringIO(reply)):
            with self.assertRaises(PlanError):
                request_plan(alignment,library,'test','https://invalid.test','DASHSCOPE_API_KEY',diagnostics_dir=root)
            r=self.records(root)
            self.assertTrue(any(x.get('response_content')=='not valid JSON' for x in r))
            self.assertTrue(any(x.get('status')=='rejected' for x in r))
    def test_report_not_claiming_true_model_error_rate(self):
        with tempfile.TemporaryDirectory() as root:
            validation(root,'object_binding',1,'rejected','NO_OBJECT_CAN_BE_USED_IN_THIS_TASK')
            report=build(root,Path(root)/'report')
            self.assertFalse(report['training_eligible']); self.assertEqual(report['logical_request_count'],0)
            self.assertIn('independent task',report['scope'])


if __name__=='__main__': unittest.main()
