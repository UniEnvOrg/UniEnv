"""UniEnvMCPServer through an in-process MCP client: tools, instances, actions, images, policy runs."""

from __future__ import annotations

import base64
import json
import sys
import time

import anyio
import numpy as np
import pytest

pytest.importorskip("mcp")

from mcp import Client  # noqa: E402

from unienv_mcp import UniEnvMCPServer  # noqa: E402

from tests_common.mcp_envs import GoalPolicy, GridEnv, make_goal_policy, make_grid  # noqa: E402


def call(server: UniEnvMCPServer, *calls, progress=None):
    """Run tool calls in one client session; returns their results."""
    async def main():
        results = []
        async with Client(server.mcp_server) as client:
            await client.list_tools()
            for name, arguments in calls:
                results.append(await client.call_tool(name, arguments, progress_callback=progress))
        return results
    return anyio.run(main)


def structured(result):
    assert not result.is_error, result.content[0].text
    return result.structured_content


def images(result):
    return [base64.b64decode(c.data) for c in result.content if c.type == "image"]


@pytest.fixture
def server():
    with UniEnvMCPServer("test") as server:
        server.add_env("grid", make_grid, dt=0.05, max_instances=2)
        server.add_policy("goto", make_goal_policy, max_steps=50, description="moves to a goal point")
        yield server


def test_tools_and_schemas(server):
    async def main():
        async with Client(server.mcp_server) as client:
            return {tool.name: tool for tool in (await client.list_tools()).tools}
    tools = anyio.run(main)
    assert set(tools) == {"grid.describe", "grid.open", "grid.reset", "grid.step", "grid.render", "grid.close",
                          "goto.run", "goto.suggest", "runs.status", "runs.cancel"}
    action = tools["grid.step"].input_schema["properties"]["action"]
    assert action["type"] == "object" and action["required"] == ["dx", "dy"]
    assert action["properties"]["dx"] == {"type": "number", "minimum": -1.0, "maximum": 1.0, "description": "units: m"}
    assert tools["grid.describe"].annotations.read_only_hint
    assert not tools["grid.step"].annotations.destructive_hint  # a simulation
    (described,) = call(server, ("grid.describe", {}))
    about = structured(described)
    assert about["step_seconds"] == 0.05 and about["policies"] == ["goto"] and not about["shared"]
    assert about["observation"]["properties"]["camera"]["description"].startswith("kind: image")


def test_open_reset_step_with_images_and_errors(server):
    results = call(
        server,
        ("grid.open", {}),
        ("grid.reset", {"instance": "grid-1", "seed": 0}),
        ("grid.step", {"instance": "grid-1", "action": {"dx": 0.5, "dy": 0.0}}),
        ("grid.step", {"instance": "grid-1", "action": {"dx": 1.5, "dy": 0.0}}),
        ("grid.step", {"instance": "grid-1", "action": {"dx": 0.5}}),
        ("grid.step", {"instance": "grid-1", "action": {"dx": 0.5, "dy": 0.0}, "repeat": 10}),
        ("grid.render", {"instance": "grid-1"}),
        ("grid.step", {"instance": "nope", "action": {"dx": 0, "dy": 0}}),
    )
    opened, reset, step, too_far, missing, repeated, render, unknown = results
    assert structured(opened)["instance"] == "grid-1"
    first = structured(reset)
    assert first["context"] == "reach the goal at (2, 0)" and first["observation"]["pos"] == {"x": -1.0, "y": 0.0}
    assert first["observation"]["camera"] == {"image": 1, "height": 16, "width": 16}
    (png,) = images(reset)
    assert png.startswith(b"\x89PNG")
    assert structured(step)["observation"]["pos"]["x"] == -0.5 and structured(step)["rewards"] == [-2.5]
    assert too_far.is_error and "dx: 1.5 is above the maximum 1" in too_far.content[0].text
    assert missing.is_error and "missing ['dy']" in missing.content[0].text
    done = structured(repeated)
    assert done["steps"] == 5 and done["terminated"] and done["info"]["success"]  # stopped at the goal
    assert "reset" in done["next"]
    assert len(images(render)) == 1
    assert unknown.is_error and "No open instance 'nope'" in unknown.content[0].text


def test_clipping_instead_of_errors():
    with UniEnvMCPServer() as server:
        server.add_env("grid", make_grid, clip_actions=True)
        _, _, step = call(server, ("grid.open", {}), ("grid.reset", {"instance": "grid-1"}),
                          ("grid.step", {"instance": "grid-1", "action": {"dx": 3.0, "dy": -3.0}}))
        assert structured(step)["observation"]["pos"] == {"x": 0.0, "y": -1.0}


def test_instances_are_limited_closed_and_expire():
    with UniEnvMCPServer() as server:
        server.add_env("grid", make_grid, max_instances=2, idle_timeout=0.5)
        first, second, third = call(server, ("grid.open", {}), ("grid.open", {}), ("grid.open", {}))
        ids = [structured(first)["instance"], structured(second)["instance"]]
        assert third.is_error and "maximum" in third.content[0].text
        instance = server._instances[ids[0]]
        env, worker_thread = instance.env, None
        (closed,) = call(server, ("grid.close", {"instance": ids[0]}))
        assert structured(closed)["closed"] and env.closed_on.startswith("unienv-mcp:grid")  # on its own thread
        expiring = server._instances[ids[1]].env
        deadline = time.monotonic() + 5
        while ids[1] in server._instances and time.monotonic() < deadline:
            time.sleep(0.1)
        assert ids[1] not in server._instances and expiring.closed_on is not None


def test_shared_env_instance():
    env = GridEnv()
    with UniEnvMCPServer() as server:
        server.add_env("robot", env, dt=0.1, physical=True)
        opened, reset, closed = call(server, ("robot.open", {}), ("robot.reset", {"instance": "robot"}),
                                     ("robot.close", {"instance": "robot"}))
        assert structured(opened) == {"instance": "robot", "shared": True, "next": "robot.reset to start an episode"}
        assert not structured(closed)["closed"] and env.closed_on is None
        tools = anyio.run(_tools, server)
        assert tools["robot.step"].annotations.destructive_hint and tools["robot.step"].annotations.open_world_hint
    assert env.closed_on is None  # shared objects belong to the caller


async def _tools(server):
    async with Client(server.mcp_server) as client:
        return {tool.name: tool for tool in (await client.list_tools()).tools}


def test_policy_runs_as_a_skill_with_progress_and_keyframes(server):
    progress = []

    async def on_progress(done, total, message):
        progress.append((done, total))

    results = call(server,
                   ("grid.open", {}), ("grid.reset", {"instance": "grid-1"}),
                   ("goto.run", {"instance": "grid-1", "keyframes": 2}),
                   ("goto.run", {"instance": "grid-1", "task": "go to (0, 1)", "max_steps": 20}),
                   progress=on_progress)
    first, second = structured(results[2]), structured(results[3])
    assert first["status"] == "finished" and first["success"] and first["steps"] == 6  # from x=-1 by 0.5 steps
    steps = [k["step"] for k in first["keyframes"]]
    assert len(steps) == 2 and steps == sorted(set(steps)) and steps[-1] == 6  # spread, ending at the last step
    assert len(images(results[2])) == 3  # two keyframes and the final camera
    # The second run continues from the goal toward the task's point (0, 1): the task replaces the context
    assert second["task"] == "go to (0, 1)" and second["observation"]["pos"] == {"x": 0.0, "y": 1.0}
    assert second["episode_steps"] == first["steps"] + second["steps"]
    policy = server._instances["grid-1"].policies["goto"]
    assert set(policy.dts) == {0.05}  # dt reached observe_transition


def test_background_runs_status_and_cancel():
    with UniEnvMCPServer() as server:
        server.add_env("slow", lambda: GridEnv(step_delay=0.02), dt=0.05)
        server.add_policy("goto", make_goal_policy, max_steps=2000)

        async def main():
            async with Client(server.mcp_server) as client:
                await client.list_tools()
                await client.call_tool("slow.open", {})
                await client.call_tool("slow.reset", {"instance": "slow-1"})
                started = await client.call_tool("goto.run", {"instance": "slow-1", "task": "go to (-1, 500)",
                                                              "max_steps": 2000, "wait": False})
                run = started.structured_content["run"]
                await anyio.sleep(0.3)
                running = await client.call_tool("runs.status", {"run": run})
                busy = await client.call_tool("slow.close", {"instance": "slow-1"})
                stepping = await client.call_tool("slow.step", {"instance": "slow-1", "action": {"dx": 0, "dy": 0}})
                await client.call_tool("runs.cancel", {"run": run})
                for _ in range(100):
                    status = await client.call_tool("runs.status", {"run": run})
                    if status.structured_content["status"] != "running":
                        break
                    await anyio.sleep(0.05)
                return running, busy, stepping, status
        running, busy, stepping, status = anyio.run(main)
        assert running.structured_content["status"] == "running" and running.structured_content["steps"] > 0
        assert busy.is_error and "busy" in busy.content[0].text
        assert stepping.is_error and "is running run-" in stepping.content[0].text
        final = structured(status)
        assert final["status"] == "cancelled" and 0 < final["steps"] < 2000


def test_suggest_does_not_act(server):
    results = call(server, ("grid.open", {}), ("grid.reset", {"instance": "grid-1"}),
                   ("goto.suggest", {"instance": "grid-1"}), ("goto.suggest", {"instance": "grid-1", "task": "go to (-1, 3)"}),
                   ("grid.step", {"instance": "grid-1", "action": {"dx": 0.0, "dy": 0.0}}))
    assert structured(results[2])["action"] == {"dx": 0.5, "dy": 0.0}
    assert structured(results[3])["action"] == {"dx": 0.0, "dy": 0.5}
    assert structured(results[4])["episode_steps"] == 1  # suggestions did not step


def test_shared_policy_runs_one_at_a_time_and_failures_are_reported():
    with UniEnvMCPServer() as server:
        server.add_env("slow", lambda: GridEnv(step_delay=0.02), dt=0.05, max_instances=2)
        server.add_policy("goto", GoalPolicy(), max_steps=100)
        results = call(server, ("slow.open", {}), ("slow.open", {}),
                       ("slow.reset", {"instance": "slow-1"}), ("slow.reset", {"instance": "slow-2"}),
                       ("goto.run", {"instance": "slow-1", "task": "go to (-1, 40)", "wait": False}),
                       ("goto.run", {"instance": "slow-2", "wait": False}),
                       ("goto.run", {"instance": "slow-1", "max_steps": 101}))
        assert results[5].is_error and "already running" in results[5].content[0].text
        assert results[6].is_error and "max_steps" in results[6].content[0].text

    def broken():
        policy = GoalPolicy()
        policy.get_action = lambda **kwargs: (_ for _ in ()).throw(RuntimeError("no weights"))
        return policy

    with UniEnvMCPServer() as server:
        server.add_env("grid", make_grid, dt=0.05)
        server.add_policy("broken", broken)
        _, _, run = call(server, ("grid.open", {}), ("grid.reset", {"instance": "grid-1"}),
                         ("broken.run", {"instance": "grid-1"}))
        outcome = structured(run)
        assert outcome["status"] == "failed" and "RuntimeError: no weights" in outcome["error"]


def test_runs_need_a_step_duration_and_a_reset():
    with UniEnvMCPServer() as server:
        server.add_env("grid", make_grid)  # no dt, and GridEnv has no control_timestep
        server.add_policy("goto", make_goal_policy)
        _, early, _, run = call(server, ("grid.open", {}), ("goto.run", {"instance": "grid-1"}),
                                ("grid.reset", {"instance": "grid-1"}), ("goto.run", {"instance": "grid-1"}))
        assert early.is_error and "Reset grid-1 first" in early.content[0].text
        assert run.is_error and "dt=" in run.content[0].text


def test_registration_checks():
    with UniEnvMCPServer() as server:
        with pytest.raises(ValueError, match="Names must be"):
            server.add_env("bad name", make_grid)
        server.add_env("grid", make_grid)
        with pytest.raises(ValueError, match="already registered"):
            server.add_policy("grid", make_goal_policy)
        with pytest.raises(ValueError, match="reserved"):
            server.add_policy("runs", make_goal_policy)
        with pytest.raises(TypeError, match="returned int"):
            server.add_env("junk", lambda: 3)


def test_bridge_exposes_a_remote_server():
    from unienv_interface.remote import RemoteServer
    from unienv_mcp import bridge

    with RemoteServer() as remote:
        remote.register("sim", GridEnv())
        remote.register_factory("lazy", make_grid, release_after=0)
        remote.register("goto", GoalPolicy())
        address = remote.listen("tcp://127.0.0.1:0")
        with bridge(address) as server:
            server._envs["sim"].dt = 0.05
            tools = anyio.run(_tools, server)
            assert {"sim.step", "lazy.step", "goto.run"} <= set(tools)
            _, reset, run = call(server, ("sim.open", {}), ("sim.reset", {"instance": "sim"}),
                                 ("goto.run", {"instance": "sim"}))
            assert structured(reset)["observation"]["pos"] == {"x": -1.0, "y": 0.0}
            assert structured(run)["success"]
        server._bridge_client.close()


def test_cli_over_stdio():
    from mcp import StdioServerParameters

    params = StdioServerParameters(command=sys.executable, args=[
        "-m", "unienv_mcp", "serve", "tests_common.mcp_envs:make_grid", "--name", "grid",
        "--policy", "tests_common.mcp_envs:make_goal_policy=goto"])

    async def main():
        async with Client(params) as client:
            names = [tool.name for tool in (await client.list_tools()).tools]
            await client.call_tool("grid.open", {})
            reset = await client.call_tool("grid.reset", {"instance": "grid-1"})
            return names, reset
    names, reset = anyio.run(main)
    assert "grid.step" in names and "goto.run" in names
    assert structured(reset)["observation"]["pos"] == {"x": -1.0, "y": 0.0}


def test_environment_failures_are_tool_errors():
    def broken_env():
        env = GridEnv()
        env.step = lambda action: (_ for _ in ()).throw(RuntimeError("simulator crashed"))
        return env

    with UniEnvMCPServer() as server:
        server.add_env("grid", broken_env)
        _, _, step = call(server, ("grid.open", {}), ("grid.reset", {"instance": "grid-1"}),
                          ("grid.step", {"instance": "grid-1", "action": {"dx": 0, "dy": 0}}))
        assert step.is_error and "grid.step failed: RuntimeError: simulator crashed" in step.content[0].text


def test_progress_polling_starts_no_threads(server):
    import threading
    before = threading.active_count()
    call(server, ("grid.open", {}), ("grid.reset", {"instance": "grid-1"}),
         ("goto.run", {"instance": "grid-1", "task": "go to (-1, 30)", "max_steps": 50}))
    assert threading.active_count() <= before + 3  # the instance worker and the sweeper, not one per poll
