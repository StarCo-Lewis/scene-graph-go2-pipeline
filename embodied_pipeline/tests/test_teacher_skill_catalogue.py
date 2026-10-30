"""Offline checks for the 2026-10-30 additive catalogue merge."""
import copy
import hashlib
import json
from pathlib import Path
import unittest

from embodied_pipeline._legacy import ROOT, load
from embodied_pipeline import instruction_skills

OUT = ROOT / 'outputs/skill_catalogue_update_20261030_v1'
def read(path): return json.loads(path.read_text(encoding='utf-8'))

class TeacherSkillCatalogueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.library=read(ROOT/'outputs/quadruped_atomic_skill_library.v0.1.json')
        cls.before=read(OUT/'library_before.json')
        cls.profiles=read(ROOT/'outputs/skill_matching_v1/skill_semantic_profiles.json')
        cls.old_profiles=read(OUT/'profiles_before.json')
        cls.manifest=read(OUT/'merge_manifest.json')

    def test_original_definitions_profiles_and_robot_profile_preserved(self):
        after={s['skill_id']:s for s in self.library['skills']}
        for old in self.before['skills']:
            self.assertEqual({k:after[old['skill_id']][k] for k in old},old)
        for sid,profile in self.old_profiles['profiles'].items():
            self.assertEqual(self.profiles['profiles'][sid],profile)
        self.assertEqual(self.library['robot_profile'],self.before['robot_profile'])
        self.assertEqual(self.library['design_principles'],self.before['design_principles'])

    def test_counts_unique_ids_and_source_coverage(self):
        ids=[s['skill_id'] for s in self.library['skills']]
        templates={t['template_id']:t for t in self.library['composite_task_templates']}
        self.assertEqual(len(ids),33);self.assertEqual(len(ids),len(set(ids)))
        self.assertEqual(len(templates),9)
        mapping=self.library['teacher_list_mapping']
        self.assertEqual(len(mapping),29)
        self.assertEqual(len({(m['source_id'],m['locator']) for m in mapping}),29)
        sources=read(OUT/'teacher_documents_extracted.json')
        source_lookup={m['source_id']:{r['locator']:r for r in source['records']}
            for m,source in zip(self.library['source_documents'],sources)}
        for m in mapping:
            record=source_lookup[m['source_id']][m['locator']]
            self.assertIn(m['source_label_zh'],record.get('text',' '.join(record.get('cells',[]))))
            self.assertTrue(set(m['skill_ids']) <= set(ids))
            self.assertTrue(set(m['template_ids']) <= set(templates))
        covered={sid for m in mapping for sid in m['skill_ids']}
        self.assertTrue(set(self.manifest['added_skill_ids']) <= covered)
        for template in templates.values():
            self.assertTrue(set(template['possible_skill_ids']) <= set(ids))
            self.assertFalse(template['execution_ready'])
        expected_humanoid={2,3,4,6,7,8,9,10,12,13,14,15}
        actual={m['locator'] for m in mapping if m['source_id']=='teacher_humanoid_20261030'}
        self.assertEqual(actual,{f'body[{i}]/paragraph' for i in expected_humanoid})

    def test_new_contracts_no_undeclared_symbols_or_fabricated_defaults(self):
        def strings(value):
            if isinstance(value,str):yield value
            elif isinstance(value,dict):
                for v in value.values():yield from strings(v)
            elif isinstance(value,list):
                for v in value:yield from strings(v)
        for s in self.library['skills']:
            params={p['name']:p for p in s['parameters']}
            self.assertEqual(len(params),len(s['parameters']))
            self.assertTrue({p['type'] for p in params.values()} <= set(self.library['parameter_types']))
            for value in strings(s['preconditions']+s['effects']):
                if value.startswith('$'):self.assertIn(value[1:],params,s['skill_id'])
            if s['skill_id'] in self.manifest['added_skill_ids']:
                self.assertTrue(s['teacher_list_links'])
                self.assertFalse(s['catalogue_metadata']['execution_ready'])
                self.assertTrue(s['required_robot_capabilities'])
                self.assertFalse(any('default' in p for p in params.values()))
                for condition in s['preconditions']+s['effects']:
                    self.assertIn(condition['predicate'],self.library['predicate_vocabulary'])
        self.assertFalse(self.library['catalogue_policy']['source_lists_exhaustive'])

    def test_enrichment_and_dependency_graph(self):
        original=copy.deepcopy(self.library)
        enriched=instruction_skills.enrich(self.library)
        self.assertEqual(self.library,original)
        self.assertEqual({s['skill_id'] for s in enriched['skills']},set(self.profiles['profiles']))
        visited=set();active=set()
        def visit(sid):
            self.assertNotIn(sid,active,'Dependency cycle')
            if sid in visited:return
            active.add(sid)
            p=self.profiles['profiles'][sid]
            self.assertTrue(p['description_en']);self.assertTrue(p['examples']);self.assertTrue(p['aliases'])
            for dependency in p['prerequisite_skills']:
                self.assertIn(dependency,self.profiles['profiles']);visit(dependency)
            active.remove(sid);visited.add(sid)
        for s in enriched['skills']:
            self.assertEqual(s['semantic_profile'],self.profiles['profiles'][s['skill_id']])
            visit(s['skill_id'])

    def test_legacy_five_skill_matrix_is_identical(self):
        from embodied_pipeline.orchestration.task_collection_worker import SKILL_IDS
        alignment=dict(status='success',context_bindings={'target':{'element_ids':[13]}},
            task_decomposition={'subtasks':[{'subtask_id':'S1','description':'Move the object.','depends_on':[]}]})
        def subset(lib):
            out=copy.deepcopy(lib);out['skills']=[s for s in out['skills'] if s['skill_id'] in SKILL_IDS];return out
        self.assertEqual(instruction_skills.make_codebook(alignment,subset(self.before)),
                         instruction_skills.make_codebook(alignment,subset(self.library)))

    def test_generic_proposal_can_reference_new_skill_but_matrix_rejects_unsupported_slots(self):
        alignment=dict(status='success',context_bindings={'target':{'element_ids':[13]}},
            task_decomposition={'subtasks':[{'subtask_id':'S1','description':'Hold the bottle steady.','depends_on':[]}]})
        proposal={'schema_version':'1.0','subtask_plans':[{'subtask_id':'S1','intent':'Maintain the existing hold.',
            'steps':[{'skill_id':'hold_and_stabilize_object','arguments':{
                'object':{'role_id':'target','element_id':13},'hand':'configured_hand',
                'constraint_reference':'provided_holding_constraints','duration_s':1.0}}]}]}
        # This is a synthetic validation fixture, never a physical command or motion default.
        plans=load('match_skills').validate_proposal(proposal,alignment,self.library)
        self.assertEqual(plans,proposal['subtask_plans'])
        restricted=copy.deepcopy(self.library)
        restricted['skills']=[s for s in restricted['skills'] if s['skill_id']=='hold_and_stabilize_object']
        with self.assertRaisesRegex(ValueError,'at most two parameter slots'):
            instruction_skills.make_codebook(alignment,restricted)

    def test_comparison_document_and_hashes(self):
        absent={'navigate_to_object','navigate_to_location','adjust_body_height','close_articulated_object',
                'pour_into_container','handover_object','inspect_object'}
        old_ids={s['skill_id'] for s in self.before['skills']}
        linked={s['skill_id'] for s in self.library['skills'] if s['teacher_list_links']}
        self.assertEqual(old_ids-linked,absent)
        text=(OUT/'existing_skills_not_in_teacher_lists.md').read_text(encoding='utf-8')
        for sid in absent:self.assertIn('`'+sid+'`',text)
        for filename,key in [('library_before.json','library_before_sha256'),('profiles_before.json','profiles_before_sha256')]:
            self.assertEqual(hashlib.sha256((OUT/filename).read_bytes()).hexdigest(),self.manifest[key])
        for path,key in [(ROOT/'outputs/quadruped_atomic_skill_library.v0.1.json','library_after_sha256'),
                         (ROOT/'outputs/skill_matching_v1/skill_semantic_profiles.json','profiles_after_sha256')]:
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),self.manifest[key])

if __name__=='__main__':unittest.main()
