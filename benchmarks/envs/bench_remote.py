"""Remote execution benchmarks: codec cost, per-step round trips and codec call counts.

Usage (from the repository root)::

    python benchmarks/envs/bench_remote.py              # codec + memory + websocket
    python benchmarks/envs/bench_remote.py --quick      # short smoke run
    python benchmarks/envs/bench_remote.py codec memory websocket count   # pick sections
    UNIENV_ROOT=/path/to/main-checkout python benchmarks/envs/bench_remote.py   # A/B

``codec`` times Codec.encode / decode of a step response; ``memory`` and ``websocket``
time one client ``env.step`` through a RemoteServer; ``count`` reports codec
encode/decode calls and bytes per step (server and client together, MemoryTransport).
Observations are a 32-float state plus an optional uint8 image.
"""
import collections
import time

import _common
from _common import bench, report, cli_args, ImageEnv, QUICK

import numpy as np

from unienv_interface.remote import Codec, RemoteClient, RemoteServer
from unienv_interface.remote import codec as codec_module

CASES = [(None, None), (None, (84, 84, 3)), (None, (224, 224, 3)), (64, (84, 84, 3))]
QUICK_CASES = [(None, None), (None, (84, 84, 3))]


def _label(img):
    return "state" if img is None else "x".join(map(str, img))


def codec_case(batch, img):
    env = ImageEnv(batch, img)
    result = env.step(np.zeros(env.action_space.shape, np.float32))
    codec = Codec()
    message = {"version": 1, "type": "response", "request_id": "x" * 32, "session_id": "y" * 32,
               "result": {"value": result, "descriptors": []}}
    payload = codec.encode(message)
    report(f"codec encode {_label(img)} ({len(payload) / 1024:.0f} KiB)", bench(lambda: codec.encode(message)), batch)
    report(f"codec decode {_label(img)}", bench(lambda: codec.decode(payload)), batch)


def remote_case(transport, batch, img):
    server = RemoteServer()
    try:
        server.register("env", ImageEnv(batch, img))
        client = server.connect() if transport == "memory" else RemoteClient.connect(server.listen())
        try:
            env = client.env("env")
            env.reset()
            action = np.zeros(env.action_space.shape, np.float32)
            env.step(action)
            us = bench(lambda: env.step(action), min_time=1.0, warmup=20)
            report(f"remote[{transport}] step {_label(img)}", us, batch)
        finally:
            client.close()
    finally:
        server.close()


def count_case(batch, img, steps=20):
    stats = collections.Counter()
    original_encode, original_decode = codec_module.Codec.encode, codec_module.Codec.decode

    def encode(self, value):
        payload = original_encode(self, value)
        stats["encode"] += 1
        stats["encode_bytes"] += len(payload)
        return payload

    def decode(self, payload):
        stats["decode"] += 1
        stats["decode_bytes"] += len(payload)
        return original_decode(self, payload)

    codec_module.Codec.encode, codec_module.Codec.decode = encode, decode
    server = RemoteServer()
    try:
        server.register("env", ImageEnv(batch, img))
        client = server.connect()
        try:
            env = client.env("env")
            env.reset()
            action = np.zeros(env.action_space.shape, np.float32)
            env.step(action)
            time.sleep(0.2)
            stats.clear()
            for _ in range(steps):
                env.step(action)
            time.sleep(0.2)  # let the server's writer thread finish
        finally:
            client.close()
    finally:
        server.close()
        codec_module.Codec.encode, codec_module.Codec.decode = original_encode, original_decode
    b = "-" if batch is None else batch
    print(f"codec calls per step {_label(img):<12} B={b:<5} encode {stats['encode'] / steps:5.1f} "
          f"({stats['encode_bytes'] / steps / 1024:8.1f} KiB)  decode {stats['decode'] / steps:5.1f} "
          f"({stats['decode_bytes'] / steps / 1024:8.1f} KiB)", flush=True)


def main():
    sections = cli_args() or ["codec", "memory", "websocket", "count"]
    cases = QUICK_CASES if QUICK else CASES
    for section in sections:
        for batch, img in cases:
            if section == "codec":
                codec_case(batch, img)
            elif section == "count":
                count_case(batch, img)
            else:
                remote_case(section, batch, img)


if __name__ == "__main__":
    main()
