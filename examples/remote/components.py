"""Run: python examples/remote/components.py (requires unienv[network])."""

import numpy as np

from unienv_interface.remote import RemoteServer, RemoteClient, RemoteWorldEnv
from demo import make_env


def main():
    with RemoteServer() as server:
        server.register("counter", make_env())
        endpoint = server.listen()
        with RemoteClient.connect(endpoint) as controller, RemoteClient.connect(endpoint) as observer:
            description = controller.describe("counter")
            world = controller.world(description["world_id"])
            node = controller.node(description["node_id"])
            env = RemoteWorldEnv(world, node)
            with observer.subscribe(node.resource_id, ["observation"]) as stream:
                env.reset()
                stream.next(timeout=2)
                for _ in range(3):
                    observation, *_ = env.step(np.ones(2, np.float32))
                    event = stream.next(timeout=2)
                    # One snapshot for the entire control step (two world steps).
                    print("Control step:", event["sequence"], "observation:", observation)
            env.close()


if __name__ == "__main__":
    main()
