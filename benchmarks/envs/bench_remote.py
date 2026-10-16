"""Remote execution benchmarks: wire encode/decode cost and env.step round trips per transport.

Usage (from the repository root)::

    python benchmarks/envs/bench_remote.py                    # all sections
    python benchmarks/envs/bench_remote.py --quick            # short smoke run
    python benchmarks/envs/bench_remote.py wire tcp unix      # pick sections
    UNIENV_ROOT=/path/to/other-checkout python benchmarks/envs/bench_remote.py   # A/B

Sections: ``wire`` (encode + decode of a step result), ``memory``, ``tcp``, ``unix``
and ``websocket`` (one client ``env.step`` through a ``RemoteServer``). Observations
are a 32-float state plus an optional uint8 image.
"""
import os
import tempfile

import _common
from _common import bench, report, cli_args, ImageEnv, QUICK

import numpy as np

from unienv_interface.remote import RemoteClient, RemoteServer, wire

CASES = [(None, None), (None, (84, 84, 3)), (64, (84, 84, 3))]
QUICK_CASES = [(None, None), (None, (84, 84, 3))]
TRANSPORTS = ("memory", "tcp", "unix", "websocket")


def _label(batch, img):
    return ("state" if img is None else "x".join(map(str, img))) + ("" if batch is None else f" B={batch}")


def wire_case(batch, img):
    env = ImageEnv(batch, img)
    message = [1, 0, env.step(np.zeros(env.action_space.shape, np.float32))]
    payload = wire.dumps(message)
    report(f"wire encode {_label(batch, img)} ({len(payload) / 1024:.0f} KiB)", bench(lambda: wire.encode(message)), batch)
    report(f"wire decode {_label(batch, img)}", bench(lambda: wire.loads(payload)), batch)


def _address(transport):
    if transport == "tcp":
        return "tcp://127.0.0.1:0"
    if transport == "unix":
        return "unix://" + os.path.join(tempfile.mkdtemp(prefix="uenv"), "bench.sock")
    return "ws://127.0.0.1:0"


def step_case(transport, batch, img, **client_options):
    with RemoteServer() as server:
        server.register("env", ImageEnv(batch, img))
        if transport == "memory":
            client = server.connect(**client_options)
        else:
            client = RemoteClient.connect(server.listen(_address(transport)), **client_options)
        with client:
            env = client.env("env")
            env.reset()
            action = np.zeros(env.action_space.shape, np.float32)
            us = bench(lambda: env.step(action), min_time=1.0, warmup=50)
    report(f"remote[{transport}] step {_label(batch, img)}", us, batch)
    return us


def main():
    sections = cli_args() or ["wire", *TRANSPORTS]
    cases = QUICK_CASES if QUICK else CASES
    for section in sections:
        for batch, img in cases:
            if section == "wire":
                wire_case(batch, img)
            else:
                step_case(section, batch, img)


if __name__ == "__main__":
    main()
