"""RemoteServer.publish: events for results the server did not produce itself."""

import numpy as np
import pytest

from unienv_interface.remote import RemoteServer

from tests_common.remote_envs import ProbeEnv


def test_publish_sends_server_side_results_to_subscribers():
    env = ProbeEnv()
    with RemoteServer() as server:
        server.register("probe", env, publish_results=False)
        server.register("other", ProbeEnv())  # publishes its remote calls, as by default
        client = server.connect()
        events, others = client.subscribe("probe"), client.subscribe("other")
        client.env("probe").step(np.zeros(1, np.float32))  # not published: the object publishes its own events
        with pytest.raises(TimeoutError):
            events.next(timeout=0.2)
        server.publish("probe", "step", env.step(np.zeros(1, np.float32)))  # e.g. a step driven on the server's side
        event = events.next(timeout=5)
        assert event.kind == "step" and event.value[1] == 2.0 and event.sequence == 1
        client.env("other").step(np.zeros(1, np.float32))
        assert others.next(timeout=5).kind == "step"
        server.publish("probe", "reset", env.reset())
        assert events.next(timeout=5).kind == "reset"
        with pytest.raises(ValueError, match="not 'render'"):
            server.publish("probe", "render", None)
        with pytest.raises(KeyError, match="No hosted environment"):
            server.publish("nope", "step", None)
        client.close()


def test_publish_without_subscribers_is_a_no_op():
    with RemoteServer() as server:
        server.register("probe", ProbeEnv())
        server.publish("probe", "step", object())  # nothing is encoded without subscribers
