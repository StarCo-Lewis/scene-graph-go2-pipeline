"""ROS is imported only for an explicit live launch. No automatic motion retries."""
import time
from .navigation_contract import GOAL_TYPES


def wait_future(future, spin, seconds):
    deadline = time.monotonic() + seconds
    while not future.done():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        spin(min(0.1, remaining))
    return True


def run_transaction(client, goal, spin, emit, serialize, result_timeout,
                    ack_timeout=15.0, cancel_timeout=10.0):
    """Injectable transport transaction. A cancel acknowledgement is not a stop."""
    handle = None
    pending = None
    result_future = None
    sent = False

    def terminal(wrapped):
        data = serialize(wrapped.result)
        ok = wrapped.status == 4 and data.get("success") is True and data.get("reason") == "arrived"
        return {"status": "server_reported_arrival" if ok else "server_reported_failure",
                "ros_goal_status": wrapped.status, "result": data,
                "robot_execution_attempted": True,
                "physical_arrival_independently_verified": False}

    try:
        emit("goal_send_started", {})
        sent = True  # A send exception can still leave delivery uncertain.
        pending = client.send_goal_async(goal, feedback_callback=lambda m: emit("feedback", serialize(m.feedback)))
        if not wait_future(pending, spin, ack_timeout):
            raise TimeoutError("Goal acceptance acknowledgement timed out")
        handle = pending.result()
        if not handle.accepted:
            return {"status": "goal_rejected", "robot_execution_attempted": True}
        emit("goal_accepted", {})
        result_future = handle.get_result_async()
        if not wait_future(result_future, spin, result_timeout):
            raise TimeoutError("Terminal result timed out")
        return terminal(result_future.result())
    except (Exception, KeyboardInterrupt) as exc:
        recovery = {"status": "execution_state_unknown_operator_check_required" if sent else "not_sent",
                    "error": type(exc).__name__ + ": " + str(exc),
                    "robot_execution_attempted": sent, "cancel_requested": False,
                    "cancel_acknowledged": False, "terminal_result_received": False}
        try:
            # A late acknowledgement may still provide the handle needed to cancel.
            if handle is None and pending is not None and wait_future(pending, spin, cancel_timeout):
                handle = pending.result()
            if handle is not None and not handle.accepted:
                recovery["status"] = "goal_rejected"
            elif handle is not None:
                result_future = result_future or handle.get_result_async()
                recovery["cancel_requested"] = True
                cancel = handle.cancel_goal_async()
                if wait_future(cancel, spin, cancel_timeout):
                    recovery["cancel_acknowledged"] = bool(cancel.result().goals_canceling)
                if wait_future(result_future, spin, cancel_timeout):
                    recovery["terminal_result_received"] = True
                    recovery["terminal"] = terminal(result_future.result())
                    recovery["status"] = "interrupted_with_server_terminal_result"
        except (Exception, KeyboardInterrupt) as cancel_exc:
            recovery["cancellation_error"] = str(cancel_exc)
        return recovery


def execute(request, emit):
    import rclpy
    from rclpy.action import ActionClient
    from nav_interfaces.action import NavigateTo
    from rosidl_runtime_py.convert import message_to_ordereddict

    if NavigateTo.Goal.get_fields_and_field_types() != GOAL_TYPES:
        raise ValueError("Installed NavigateTo Goal schema does not match the reviewed interface")
    rclpy.init(args=[])
    node = None
    try:
        node = rclpy.create_node("embodied_navigation_replay")
        client = ActionClient(node, NavigateTo, request["action_name"])
        if not client.wait_for_server(timeout_sec=15.0):
            raise TimeoutError("ROS2 Action server unavailable; no goal sent")
        goal = NavigateTo.Goal()
        for key, value in request["goal"].items():
            setattr(goal, key, value)
        return run_transaction(client, goal,
                               lambda seconds: rclpy.spin_once(node, timeout_sec=seconds),
                               emit, message_to_ordereddict, request["goal"]["timeout"] + 15.0)
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.shutdown()
