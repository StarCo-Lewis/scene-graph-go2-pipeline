"""Feedback flow extracted from the legacy runner without changing policies."""

import copy
import logging
import time

from .. import perception, execution, errors, memory, validation
from .._legacy import load
from ..contracts import EpisodeServices
from .proposal import propose_once


def _post_rollout_audit(args, folder, summary, state, episode, auditor, compact):
    """Private authored diagnostics remain after rollout and never gate commits."""
    if args.preflight_only:
        return
    audit_rows = []
    for row in summary["rounds"]:
        planning = folder / row["planning_path"]
        if "output_memory_version" not in row:
            continue
        begin = memory.read_json(planning / "runtime_state.json")
        end = memory.read_json(planning.parent / "memory_after_feedback.json")
        proposed = planning / "skill_plan_proposed.json"
        plan = memory.read_json(proposed) if proposed.is_file() else {"subtask_plans": []}
        try:
            diagnostic = auditor(row["round"], begin, end, plan, episode["episode_id"])
            diagnostic["proposed_placement_events"] = diagnostic.pop("ordered_placement_events")
            diagnostic["scope"] = "Post-rollout private checks; proposed events are NOT executed events"
            feedback = memory.read_json(planning.parent / "execution_feedback.json")
            diagnostic["successful_proxy_placement_events"] = [
                [t["arguments"]["object"]["element_id"], t["arguments"]["surface"]["element_id"]]
                for t in feedback["trace"] if t["skill_id"] == "place_object_on_surface"
                and t["status"] == "proxy_action_succeeded"]
            successful_plan = {"subtask_plans": [{"steps": [t for t in feedback["trace"]
                               if t["status"] == "proxy_action_succeeded"]}]}
            actual = auditor(row["round"], begin, end, successful_plan, episode["episode_id"])
            diagnostic["reference_checks"] = actual["reference_checks"]
            diagnostic["reference_checks_all_passed"] = actual["reference_checks_all_passed"]
            process_check = feedback.get("process_constraint_evaluation")
            if process_check is not None:
                diagnostic["process_constraint_evaluation"] = process_check
                diagnostic["reference_checks"]["declared_process_constraints_passed"] = process_check["passed"]
                diagnostic["reference_checks_all_passed"] = (
                    diagnostic["reference_checks_all_passed"] and process_check["passed"])
            for key in ("navigation_visits", "strict_id_visit_sequence_matched"):
                if key in actual:
                    diagnostic["proposed_" + key] = diagnostic[key]
                    diagnostic[key] = actual[key]
        except (ValueError, KeyError, TypeError) as exc:
            diagnostic = dict(status="audit_error", error=str(exc))
        memory.write_json(planning.parent / "evaluation/post_rollout_diagnostics.json", diagnostic)
        audit_rows.append(dict(round=row["round"], execution_status=row["status"], **diagnostic))
    memory.write_json(folder / "evaluation/final_audit.json", dict(rows=audit_rows,
        final_state=compact(state), semantic_success_rate=None,
        scope="Authored proxy reference diagnostics, not independent intent or physical truth"))


def run_episode_feedback(args, episode, folder, encoder, *, services=None):
    """Default calls preserve the old flow; explicit services enable offline tests."""
    services = services or EpisodeServices()
    scene_loader = services.scene_loader or perception.scene_fixture
    proposer = services.proposer or propose_once
    executor = services.executor or execution.execute_proxy
    cfg, nodes, edges, hashes = scene_loader(episode["scene"])
    tracked = tuple(episode.get("tracked_object_ids", (86, 121)))
    compact = lambda value: memory.compact_state(value, tracked)
    auditor = getattr(args, "feedback_evaluator", load("dialogue_chain_cases").evaluate)
    if "user_location" in episode:
        cfg["conversation_context"]["user_location"] = copy.deepcopy(episode["user_location"])
        cfg["conversation_context"]["user_activity"] = episode["user_activity"]
        validation.conversation_context(cfg, nodes)
    cfg.update(case_id="feedback_" + episode["episode_id"], inference_mode=args.inference_mode,
               inference_experiment=True)
    state = memory.initial_state(nodes, cfg)
    history = []
    summary = dict(episode_id=episode["episode_id"], scene=episode["scene"],
        **load("dialogue_execution_feedback").wire_profile(args),
        user_location=copy.deepcopy(cfg["conversation_context"]["user_location"]),
        user_activity=cfg["conversation_context"]["user_activity"],
        status="preparing", planned_round_count=len(episode["rounds"]), rounds=[],
        processed_round_count=0, fresh_llm_calls=False, source_files=hashes,
        later_rounds_skipped=0, simulation_executed=False, physical_grasp_verified=False,
        semantic_success_rate=None, state_commit_scope="proxy_execution_feedback_not_physical")
    memory.write_json(folder / "initial_symbolic_state.json", state)
    memory.write_json(folder / "source_snapshot/scene_graph.json", nodes)
    memory.write_json(folder / "source_snapshot/scene_graph_edges.json", edges)
    memory.write_json(folder / "source_snapshot/fixture.json", cfg)
    memory.write_json(folder / "memory_checkpoint.json", dict(state=state, history=history))
    started = time.monotonic()
    destination = folder / "episode_summary.json"
    try:
        for descriptor in episode["rounds"]:
            n = descriptor["round"]
            planning = folder / f"round_{n}/planning"
            perception.prepare_feedback(planning, descriptor, state, history, cfg, nodes, edges, args, compact)
            row = dict(round=n, instruction=descriptor["instruction"], status="input_prepared",
                input_memory_version=state["backend"]["memory_version"], history_round_count=len(history),
                begin_state=compact(state), planning_path=str(planning.relative_to(folder)))
            summary["rounds"].append(row)
            if args.preflight_only:
                row["status"] = "first_round_input_preflight_passed"
                summary["status"] = "preflight_completed_first_round_only"
                break
            row["status"] = "proposing"
            summary.update(status="running", fresh_llm_calls=True)
            memory.write_json(destination, summary)
            logging.info("START feedback dialogue=%s round=%s", episode["episode_id"], n)
            proposal, plan, alignment = proposer(args, planning, encoder)
            if proposal["status"] == "infrastructure_error":
                raise RuntimeError(proposal["error"])
            if memory.read_json(planning / "runtime_state.json") != state:
                raise ValueError("Input memory mutated by proposer; experiment integrity failure")
            if plan is None:
                result = errors.unavailable_result(state, proposal.get("error"))
            else:
                result = executor(plan, alignment, memory.read_json(planning / "execution_skill_library.json"),
                                  memory.read_json(planning / "input_memory/scene/scene_graph.json"), state, compact)
            memory.write_json(planning.parent / "execution_feedback.json", result)
            state, entry = memory.commit_feedback(folder, planning, descriptor, state, history,
                                                 result, cfg, nodes, compact)
            history.append(entry)
            memory.write_json(planning.parent / "memory_after_feedback.json", state)
            row.update(status=result["status"], error=result["error"], output_memory_version=n,
                end_state=compact(state), proposed_action_count=result["proposed_action_count"],
                attempted_action_count=result["attempted_action_count"],
                successful_action_count=result["successful_action_count"],
                skipped_action_count=result["skipped_action_count"],
                goal_success="not_evaluated_during_rollout")
            memory.write_json(destination, summary)
            logging.info("END feedback dialogue=%s round=%s status=%s; next user turn continues",
                         episode["episode_id"], n, row["status"])
        else:
            summary["status"] = "dialogue_rollout_completed_not_success_claim"
        _post_rollout_audit(args, folder, summary, state, episode, auditor, compact)
    except KeyboardInterrupt:
        summary["status"] = "interrupted"
    except Exception as exc:
        summary.update(status="harness_error", error=str(exc))
        logging.exception("Feedback rollout integrity/infrastructure error")
    finally:
        summary.update(duration_seconds=time.monotonic() - started, processed_round_count=len(history),
            successful_execution_round_count=sum(e["execution_status"] == "proxy_actions_completed" for e in history),
            failed_round_count=sum(e["execution_status"] != "proxy_actions_completed" for e in history),
            first_error_round=next((e["round"] for e in history if e["error"]), None),
            last_committed_memory_version=state["backend"]["memory_version"],
            proposed_action_count=sum(r.get("proposed_action_count", 0) for r in summary["rounds"]),
            attempted_action_count=sum(r.get("attempted_action_count", 0) for r in summary["rounds"]),
            successful_action_count=sum(r.get("successful_action_count", 0) for r in summary["rounds"]),
            later_rounds_skipped=len(episode["rounds"]) - len(summary["rounds"]))
        memory.write_json(destination, summary)
    return summary

