"""Offline configuration tests, not live Bullet calibration evidence."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

from stage_collision_config import prepare, verify_loaded


class StageConfigTests(unittest.TestCase):
    def source(self, root, **extra):
        assets=Path(root)/'assets'; assets.mkdir()
        mesh=assets/'stage.glb'; mesh.write_bytes(b'fake geometry for offline config test')
        source=assets/'original.stage_config.json'
        source.write_text(json.dumps({'render_asset':'stage.glb','collision_asset':'stage.glb',**extra}))
        return source,mesh

    def test_new_run_explicit_margin_preserves_source_and_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            source,mesh=self.source(directory); before=source.read_bytes(); geometry=mesh.read_bytes()
            config,report=prepare(source,Path(directory)/'run')
            data=json.loads(config.read_text())
            self.assertEqual(data['margin'],.001); self.assertTrue(data['is_collidable'])
            self.assertEqual(Path(data['collision_asset']),mesh.resolve())
            self.assertEqual(source.read_bytes(),before); self.assertEqual(mesh.read_bytes(),geometry)
            self.assertIsNone(report['source_config_margin'])
            self.assertFalse(report['contact_rejection_tolerance_modified'])
            sim=NS(get_stage_initialization_template=lambda:NS(margin=.001), get_stage_is_collidable=lambda:True)
            verify_loaded(sim,report,Path(directory)/'run')
            self.assertEqual(report['status'],'loaded_parameters_verified')

    def test_stage_default_not_silently_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            source,_=self.source(directory); _,report=prepare(source,Path(directory)/'run')
            sim=NS(get_stage_initialization_template=lambda:NS(margin=.04), get_stage_is_collidable=lambda:True)
            with self.assertRaisesRegex(ValueError,'MISMATCH'): verify_loaded(sim,report,Path(directory)/'run')
            self.assertEqual(report['loaded_stage_margin_m'],.04)
            saved=json.loads((Path(directory)/'run/stage_collision_configuration.json').read_text())
            self.assertEqual(saved['status'],'loaded_parameters_rejected')

    def test_disabled_stage_collisions_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source,_=self.source(directory); _,report=prepare(source,Path(directory)/'run')
            sim=NS(get_stage_initialization_template=lambda:NS(margin=.001), get_stage_is_collidable=lambda:False)
            with self.assertRaisesRegex(ValueError,'MISMATCH'): verify_loaded(sim,report,Path(directory)/'run')

    def test_missing_verification_api_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            source,_=self.source(directory); _,report=prepare(source,Path(directory)/'run')
            with self.assertRaises(ValueError): verify_loaded(NS(),report,Path(directory)/'run')
            self.assertEqual(report['status'],'loaded_parameters_rejected')

    def test_existing_output_and_missing_geometry_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source,mesh=self.source(directory); prepare(source,Path(directory)/'run')
            with self.assertRaisesRegex(ValueError,'overwrite'): prepare(source,Path(directory)/'run')
            # Refer to a never-existing asset; do not modify or delete source files.
            source2=source.parent/'missing.stage_config.json'
            source2.write_text(json.dumps({'render_asset':'absent.glb','collision_asset':'absent.glb'}))
            with self.assertRaisesRegex(ValueError,'Missing stage asset'): prepare(source2,Path(directory)/'run2')

    def test_nonmeter_stage_and_disabled_source_collisions_rejected(self):
        for settings in ({'units_to_meters':.01}, {'is_collidable':False}):
            with self.subTest(settings=settings), tempfile.TemporaryDirectory() as directory:
                source,_=self.source(directory,**settings)
                with self.assertRaises(ValueError): prepare(source,Path(directory)/'run')


if __name__=='__main__': unittest.main()
