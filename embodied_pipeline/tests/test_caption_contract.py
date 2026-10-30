"""Offline regressions: multi-view arrays, bounded retries, and verified cache reuse."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from embodied_pipeline.perception.caption_contract import request_caption, validate_caption
from embodied_pipeline.perception.caption_cache import reuse_captions
from embodied_pipeline.perception.real_rgbd import write, sha


def caption():
    return dict(caption="Printer front panel", possible_tags=["printer panel"],
        object_tag="printer panel", entity_type="object_part", is_complete_object=False,
        parent_object_tag="printer", confidence=0.95)


def envelope(value, request_id="offline-response"):
    return dict(id=request_id, choices=[dict(message={"content": json.dumps(value)}, finish_reason="stop")],
                usage={"prompt_tokens": 400, "completion_tokens": 100, "total_tokens": 500})


class CaptionTests(unittest.TestCase):
    def test_array_rejected_then_regenerated_with_evidence_and_usage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            payloads = []
            responses = iter([envelope([caption()] * 3), envelope(caption())])
            def transport(req, **kwargs):
                payloads.append(json.loads(req.data))
                value = next(responses)
                kwargs["audit"].received(1, json.dumps(value))
                return value
            options = SimpleNamespace(model="fixture", base_url="https://invalid.test")
            with patch("urllib.request.urlopen", side_effect=AssertionError("Network forbidden")):
                _, value, attempts = request_caption(options, "fake-secret-key", [], root, transport=transport)
            self.assertEqual(value, caption())
            self.assertEqual(attempts, 2)
            self.assertIn("not list", payloads[1]["messages"][0]["content"][-1]["text"])
            parses = [json.loads(p.read_text()) for p in (root / "llm_response_audit").glob("*_parse.json")]
            self.assertEqual({p["status"] for p in parses}, {"validated", "rejected"})
            ledger = json.loads(next((root / "caption_retry_audit").glob("*.json")).read_text())
            self.assertEqual([a["provider_usage"]["total_tokens"] for a in ledger["attempts"]], [500, 500])
            self.assertNotIn("fake-secret-key", "".join(p.read_text() for p in root.rglob("*.json")))

    def test_format_attempts_are_bounded_and_network_not_semantically_retried(self):
        for failure in ("format", "transport"):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as tmp:
                calls = []
                def transport(req, **kwargs):
                    calls.append(req)
                    if failure == "transport":
                        kwargs["audit"].transport_error(1, TimeoutError("fixture"))
                        raise TimeoutError("fixture")
                    value = envelope([caption()])
                    kwargs["audit"].received(1, json.dumps(value))
                    return value
                with self.assertRaises((ValueError, TimeoutError)):
                    request_caption(SimpleNamespace(model="fixture", base_url="https://invalid.test"),
                                    "fake-secret", [], tmp, transport=transport)
                self.assertEqual(len(calls), 3 if failure == "format" else 1)
                ledger = json.loads(next((Path(tmp) / "caption_retry_audit").glob("*.json")).read_text())
                self.assertEqual(ledger["status"], "format_exhausted" if failure == "format" else "transport_failed")
                if failure == "transport":
                    self.assertIsNone(ledger["attempts"][0]["provider_usage"])

    def test_precise_errors_and_field_validation(self):
        with self.assertRaisesRegex(ValueError, "not list"):
            validate_caption([caption(), caption(), caption()])
        with self.assertRaisesRegex(ValueError, "Missing caption fields"):
            validate_caption({})
        for key, value in (("confidence", True), ("confidence", float("nan")), ("entity_type", []),
                           ("possible_tags", "printer"), ("is_complete_object", 1)):
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_caption({**caption(), key: value})
        with self.assertRaisesRegex(ValueError, "truncated"):
            validate_caption(caption(), "length")

    def cache_fixture(self, root):
        summary = root / "old/run_summary.json"
        captions = summary.parent / "captions"
        crop = captions / "crops/object_000/view.jpg"
        crop.parent.mkdir(parents=True)
        crop.write_bytes(b"offline crop fixture")
        checkpoint = dict(summary_sha256="original-hash", map_sha256="map-hash", map_file="map-file")
        hashes = {"upstream": "source-hash"}
        write(summary, dict(schema_version="real_scene_semantic_continuation_v1", status="failed_or_interrupted",
            current_stage="captions", map_unchanged=True, upstream_source_unchanged=True,
            resumed_from=checkpoint, semantic_model="fixture", base_url="https://invalid.test", source_code_hashes=hashes))
        raw = envelope(caption())
        record = dict(id=0, status="SUCCESS", semantic_schema_version=2, model="fixture", qwen=caption(),
            captions=[caption()["caption"]], request_id=raw["id"], usage=raw["usage"], selected_views=[{"crop": str(crop)}])
        write(captions / "objects/object_000.json", record)
        write(captions / "llm_response_audit/caption_fixture_http_1.json",
              dict(http_status=200, response_body=json.dumps(raw)))
        return summary, checkpoint, hashes

    def test_cache_copies_verified_successes_preserving_old_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            summary, checkpoint, hashes = self.cache_fixture(root)
            original = {p: sha(p) for p in summary.parent.rglob("*") if p.is_file()}
            dest = root / "new/captions"
            result = reuse_captions(summary, dest, checkpoint, "fixture", "https://invalid.test", hashes)
            self.assertEqual(result["reused_count"], 1)
            self.assertEqual(sha(dest / "objects/object_000.json"), original[summary.parent / "captions/objects/object_000.json"])
            self.assertTrue(all(sha(p) == digest for p, digest in original.items()))
            with self.assertRaises(FileExistsError):
                reuse_captions(summary, dest, checkpoint, "fixture", "https://invalid.test", hashes)

    def test_cache_rejects_mismatched_map_model_evidence_or_missing_crop(self):
        for mutation in ("map", "model", "record", "crop", "audit"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                summary, checkpoint, hashes = self.cache_fixture(root)
                model = "fixture"
                if mutation == "map":
                    checkpoint["map_sha256"] = "different-map"
                elif mutation == "model":
                    model = "different-model"
                elif mutation == "record":
                    path = summary.parent / "captions/objects/object_000.json"
                    record = json.loads(path.read_text())
                    record["qwen"]["confidence"] = 0.1
                    write(path, record)
                elif mutation == "crop":
                    (summary.parent / "captions/crops/object_000/view.jpg").unlink()
                else:
                    (summary.parent / "captions/llm_response_audit/caption_fixture_http_1.json").unlink()
                with self.assertRaises(ValueError):
                    reuse_captions(summary, root / "new/captions", checkpoint, model, "https://invalid.test", hashes)
                self.assertFalse((root / "new/captions").exists())


if __name__ == "__main__":
    unittest.main()
