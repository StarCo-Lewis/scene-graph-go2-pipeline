"""Offline regression checks for English-only active skill metadata."""
import json
from pathlib import Path
import unittest

from match_skills import compact_skills
from skill_retrieval import enrich


class EnglishSkillMetadataTests(unittest.TestCase):
    def test_all_skill_metadata_and_model_payloads_are_english(self):
        here = Path(__file__).resolve().parent
        library_text = (here.parent / 'quadruped_atomic_skill_library.v0.1.json').read_text(encoding='utf-8')
        profiles_text = (here / 'skill_semantic_profiles.json').read_text(encoding='utf-8')
        self.assertTrue(library_text.isascii())
        self.assertTrue(profiles_text.isascii())
        library = json.loads(library_text)
        self.assertEqual(len(library['skills']), 17)
        for skill in library['skills']:
            self.assertTrue(skill['description_en'].strip())
            self.assertNotIn('description_zh', skill)
        for row in compact_skills(library):
            self.assertTrue(row['description'].strip())
        enriched = enrich(library)
        for row in compact_skills(enriched):
            self.assertTrue(row['description'].strip())
            self.assertTrue(json.dumps(row, ensure_ascii=False).isascii())


if __name__ == '__main__':
    unittest.main()
