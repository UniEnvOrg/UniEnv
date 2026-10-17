"""Compose a remote world and node with a local WorldEnv (one remote call per hook).

Run: python examples/remote/components.py
"""

import numpy as np

from unienv_interface.remote import RemoteClient, RemoteServer
from unienv_interface.world import WorldEnv
from demo import make_env


def main() -> None:
    with RemoteServer() as server:
        server.register_recursive("counter", make_env())  # also exposes counter/world and counter/node
        with RemoteClient.connect(server.listen("tcp://127.0.0.1:0")) as client:
            print("Resources:", sorted(client.resources))
            env = WorldEnv(client.world("counter/world"), client.node("counter/node"))
            env.reset()
            for _ in range(3):
                # set_next_action, two world steps (control 0.02 s / world 0.01 s), the
                # node's step hooks and the getters each run remotely.
                observation, reward, *_ = env.step(np.ones(2, np.float32))
                print("Observation:", observation, "reward:", reward)


if __name__ == "__main__":
    main()
