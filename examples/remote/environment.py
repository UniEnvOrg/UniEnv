"""Run: python examples/remote/environment.py (requires unienv[network])."""

import numpy as np

from unienv_interface.remote import RemoteServer, RemoteClient
from demo import make_env


def main():
    with RemoteServer() as server:
        server.register("counter", make_env())
        endpoint = server.listen()
        print("Serving:", endpoint)
        with RemoteClient.connect(endpoint) as controller, RemoteClient.connect(endpoint) as observer:
            env = controller.env("counter")
            with observer.subscribe("counter", ["observation", "reward"]) as stream:
                print("Reset:", env.reset(seed=0))
                print("Snapshot:", stream.next(timeout=2))
                for _ in range(3):
                    print("Step:", env.step(np.ones(2, np.float32)))
                    print("Observed sequence:", stream.next(timeout=2)["sequence"])


if __name__ == "__main__":
    main()
