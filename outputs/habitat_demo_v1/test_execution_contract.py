import copy
import unittest
from run_sideboard_full_pipeline import build_execution_contract, scoped_skill_library
from run_manipulation_proxy import EXPECTED_SKILLS


class ContractTests(unittest.TestCase):
    def test_manipulation_inventory_matches_executor_without_mutating_library(self):
        library = {"skills": [{"skill_id": sid} for sid in set(EXPECTED_SKILLS)]
                   + [{"skill_id": "inspect_object"}]}
        original = copy.deepcopy(library)
        contract = build_execution_contract("kettle_to_sideboard")
        scoped = scoped_skill_library(library, contract)
        self.assertEqual(contract["supported_ordered_sequence"], EXPECTED_SKILLS)
        self.assertNotIn("inspect_object", [s["skill_id"] for s in scoped["skills"]])
        self.assertEqual(library, original)
        self.assertIn("explicit user", contract["unsupported_requirement_policy"])

    def test_navigation_inventory_and_missing_skill(self):
        contract = build_execution_contract("navigation")
        self.assertEqual(contract["supported_skill_ids"], ["navigate_to_object"])
        with self.assertRaises(ValueError):
            scoped_skill_library({"skills": []}, contract)
