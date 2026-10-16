"""Host an env and drive it from a client while another client observes it.

Run: python examples/remote/environment.py [tcp://127.0.0.1:0 | unix:///tmp/unienv.sock | ws://127.0.0.1:0]

In practice the server runs in its own process (``server.serve(address)``) and the
clients connect to it with ``RemoteClient.connect(address)``.
"""

import sys

import numpy as np

from unienv_interface.remote import RemoteClient, RemoteServer
from demo import make_env


def main(address: str = "tcp://127.0.0.1:0") -> None:
    with RemoteServer() as server:
        server.register("counter", make_env())
        address = server.listen(address)
        print("Serving on", address)
        with RemoteClient.connect(address) as controller, RemoteClient.connect(address) as observer:
            env = controller.env("counter")
            print("Observation space:", env.observation_space)
            with observer.subscribe("counter") as stream:
                print("Reset:", env.reset(seed=0))
                for _ in range(3):
                    observation, reward, *_ = env.step(np.ones(2, np.float32))
                    event = stream.next(timeout=2)  # latest result, whoever stepped
                    print(f"Step: obs={observation} reward={reward}; observer saw {event.kind} #{event.sequence}")


if __name__ == "__main__":
    main(*sys.argv[1:])
