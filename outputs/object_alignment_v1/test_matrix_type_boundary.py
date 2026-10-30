import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
import urllib.error
from unittest.mock import patch

from align_objects import call_llm, AlignmentError
from llm_graph_binding import request_bindings, request_validated_bindings, BindingError
import test_structural_matrix as fixtures


@patch.dict(os.environ,{'OFFLINE_MATRIX_KEY':'dummy'})
class MatrixTypeBoundaryTests(unittest.TestCase):
    def response(self,data):
        return dict(choices=[dict(message=dict(content=json.dumps(data)),finish_reason='stop')])

    def call(self,raw,mode,temp):
        with patch('align_objects.request_json',return_value=self.response(raw)) as request:
            value=call_llm(raw['instruction'],'offline','https://invalid.test','OFFLINE_MATRIX_KEY',
                max_attempts=1,decomposition_format='matrix',matrix_contract=mode,diagnostics_dir=Path(temp))
            return value,json.loads(request.call_args.args[0].data)

    def test_strict_decomposition_request_is_actually_schema_constrained(self):
        with tempfile.TemporaryDirectory() as temp:
            result,payload=self.call(fixtures.sample(),'strict_schema',temp)
            self.assertEqual(payload['response_format']['type'],'json_schema')
            self.assertTrue(payload['response_format']['json_schema']['strict'])
            self.assertEqual(result['matrix_type_contract']['mode'],'strict_schema')
            self.assertFalse(result['matrix_type_contract']['normalized'])

    def test_strict_provider_type_violation_is_not_silently_repaired(self):
        raw=fixtures.sample(); raw['roles'][0][0]='1'
        with tempfile.TemporaryDirectory() as temp, patch('align_objects.request_json',return_value=self.response(raw)) as request:
            with self.assertRaisesRegex(AlignmentError,'MATRIX_TYPE_CONTRACT_REJECTED'):
                call_llm(raw['instruction'],'offline','https://invalid.test','OFFLINE_MATRIX_KEY',max_attempts=1,
                    decomposition_format='matrix',matrix_contract='strict_schema',diagnostics_dir=Path(temp))
            self.assertEqual(request.call_count,1)
            evidence=json.loads((Path(temp)/'llm_decomposition_attempt_1.json').read_text())
            self.assertEqual(json.loads(evidence['response_content'])['roles'][0][0],'1')
            self.assertFalse((Path(temp)/'llm_decomposition_attempt_1.typed.json').exists())
            record=json.loads((Path(temp)/'llm_decomposition_attempt_1.type_contract.json').read_text())
            self.assertEqual(record['status'],'rejected')

    def test_compatibility_accepts_only_known_numeric_columns_and_records_it(self):
        raw=fixtures.sample(); raw['roles'][0][0]='1'; raw['roles'][0][3]='1'
        raw['roles'][0][1]='1'
        with tempfile.TemporaryDirectory() as temp:
            result,payload=self.call(raw,'typed_json',temp)
            self.assertEqual(payload['response_format'],{'type':'json_object'})
            self.assertTrue(result['matrix_type_contract']['normalized'])
            self.assertEqual(result['task_roles'][0]['name'],'1')
            evidence=json.loads((Path(temp)/'llm_decomposition_attempt_1.json').read_text())
            self.assertEqual(json.loads(evidence['response_content'])['roles'][0][0],'1')
            normalized=json.loads((Path(temp)/'llm_decomposition_attempt_1.typed.json').read_text())
            self.assertEqual(normalized['roles'][0][0],1)

    def test_compatibility_does_not_resolve_undefined_references(self):
        raw=fixtures.sample(); raw['task_roles'][0][1]='999'
        with tempfile.TemporaryDirectory() as temp, self.assertRaisesRegex(AlignmentError,'Undefined role code 999'):
            self.call(raw,'typed_json',temp)

    def test_schema_unsupported_http_error_does_not_retry_in_json_object(self):
        error=urllib.error.HTTPError('https://invalid.test',400,'unsupported',{},io.BytesIO(b'json_schema unsupported'))
        with patch('align_objects.request_json',side_effect=error) as request:
            with self.assertRaisesRegex(AlignmentError,'HTTP 400'):
                call_llm('Visit a chair.','offline','https://invalid.test','OFFLINE_MATRIX_KEY',
                    max_attempts=3,decomposition_format='matrix',matrix_contract='strict_schema')
            self.assertEqual(request.call_count,1)
            self.assertEqual(json.loads(request.call_args.args[0].data)['response_format']['type'],'json_schema')

    def fixture(self):
        from align_objects import validate_decomposition
        from structural_matrix import from_decomposition_matrix
        roles=validate_decomposition(from_decomposition_matrix(fixtures.sample()))['task_roles']
        nodes=fixtures.inventory(); raw,book=fixtures.binding(roles,nodes)
        return roles,nodes,raw,book

    def test_binding_request_schema_is_bound_to_current_program_codebook(self):
        roles,nodes,raw,book=self.fixture()
        with patch('llm_graph_binding.request_json',return_value=self.response(raw)) as request:
            self.assertEqual(request_bindings(roles,nodes,'offline','https://invalid.test','OFFLINE_MATRIX_KEY',
                output_format='matrix',matrix_contract='strict_schema'),raw)
            fmt=json.loads(request.call_args.args[0].data)['response_format']
            props=fmt['json_schema']['schema']['properties']
            self.assertEqual(fmt['type'],'json_schema')
            self.assertEqual(props['codebook_sha256']['const'],book['codebook_sha256'])
            self.assertEqual(props['matches']['items']['prefixItems'][1]['enum'],book['active_scene_ids'])

    def test_binding_compatibility_preserves_raw_and_ambiguity(self):
        roles,nodes,raw,_=self.fixture()
        raw['decisions'][0]=['1','3']; raw['matches']=raw['matches'][1:]; raw['candidates']=[['1','152']]
        with tempfile.TemporaryDirectory() as temp, patch('llm_graph_binding.request_bindings',return_value=raw) as request:
            decoded,bound,errors,_=request_validated_bindings(roles,nodes,'offline','invalid','KEY',
                output_format='matrix',matrix_contract='typed_json',max_validation_attempts=1,diagnostics_dir=Path(temp))
            self.assertEqual(request.call_count,1)
            self.assertNotIn('role_001',bound)
            self.assertEqual(errors[0]['reason'],'ambiguous')
            saved=json.loads((Path(temp)/'llm_binding_matrix_proposal_attempt_1.json').read_text())
            self.assertEqual(saved['candidates'],[['1','152']])
            self.assertEqual(decoded['bindings'][0]['element_ids'],[152])

    def test_binding_strict_rejects_quote_without_second_proposal(self):
        roles,nodes,raw,_=self.fixture(); raw['matches'][0][1]='152'
        with tempfile.TemporaryDirectory() as temp, patch('llm_graph_binding.request_bindings',return_value=raw) as request:
            with self.assertRaises(BindingError):
                request_validated_bindings(roles,nodes,'offline','invalid','KEY',output_format='matrix',
                    matrix_contract='strict_schema',max_validation_attempts=1,diagnostics_dir=Path(temp))
            self.assertEqual(request.call_count,1)
            self.assertTrue((Path(temp)/'llm_binding_matrix_proposal_attempt_1.json').exists())
            self.assertFalse((Path(temp)/'llm_binding_proposal.json').exists())


if __name__=='__main__': unittest.main()
