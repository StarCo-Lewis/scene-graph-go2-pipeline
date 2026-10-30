"""Offline automatic queue review regressions."""
import tkinter as tk
import unittest
from unittest.mock import Mock
from embodied_pipeline._legacy import ROOT
from embodied_pipeline.tests import test_auto_navigation as fixtures
from embodied_pipeline.execution.navigation_contract import sha
from embodied_pipeline.ui.multi_navigation import MultiNavigationPanel


class AutomaticQueueTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.AutoMappingTests();self.fixture.setUp();self.addCleanup(self.fixture.doCleanups);self.fixture.fill_start()
        self.panel=self.fixture.panel; c=self.panel.console;c.controls=self.panel
        c.loaded_plan_path=ROOT/'outputs/real_scene_runs/nav74_123_reviewed_20261030/planning/engineering_plan.json'
        c.loaded_plan_hash=sha(c.loaded_plan_path)
        c.navigation_view=Mock()
        m=self.multi=object.__new__(MultiNavigationPanel);m.console=c;c.multi_navigation=m
        m.route=m.route_key=m.observed_plan=m.auto_attempt=m.route_start=None
        m.table=Mock();m.table.get_children.return_value=[]
        m.execute_button=Mock();m.after=Mock()
        m.reviewed=tk.BooleanVar(master=self.fixture.tcl,value=False)
        m.route_note=tk.StringVar(master=self.fixture.tcl)

    def test_completed_plan_populates_all_rows_and_map_without_dispatch(self):
        m=self.multi;m.refresh()
        self.assertEqual([r['target_element_id'] for r in m.route],[74,123])
        self.assertEqual(m.table.insert.call_count,2)
        self.assertFalse(m.reviewed.get());self.assertFalse(self.panel.active)
        self.assertEqual(len(self.panel.console.navigation_view.preview_goals.call_args.args[0]),2)
        # Polling preserves a manually edited candidate.
        m.route[0]['goal']['x']+=.1;m.reviewed.set(True);old=m.route
        m.refresh();self.assertIs(m.route,old);self.assertTrue(m.reviewed.get())

    def test_missing_start_waits_then_distance_change_regenerates_queue(self):
        self.panel.start['x'].set('');self.multi.refresh();self.assertIsNone(self.multi.route)
        self.panel.start['x'].set('0');self.multi.refresh();old=self.multi.route
        self.multi.reviewed.set(True);self.panel.clearance_m=.5;self.multi.refresh()
        self.assertIsNot(self.multi.route,old);self.assertFalse(self.multi.reviewed.get())
        self.assertNotEqual(self.multi.route[0]['goal'],old[0]['goal'])

