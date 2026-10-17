"""Remote execution benchmarks: wire encode/decode cost and env.step round trips per transport.

Usage (from the repository root)::

    python benchmarks/envs/bench_remote.py                    # all sections
    python benchmarks/envs/bench_remote.py --quick            # short smoke run
    python benchmarks/envs/bench_remote.py wire tcp unix      # pick sections
    UNIENV_ROOT=/path/to/other-checkout python benchmarks/envs/bench_remote.py   # A/B

Sections: ``wire`` (encode + decode of a step result), ``jpeg`` (JPEG encode + decode
of the image), ``memory``, ``tcp``, ``unix`` and ``websocket`` (one client ``env.step``
through a ``RemoteServer``, raw and with ``compress={"image": "jpeg:90"}``).
Observations are a 32-float state plus an optional smooth, camera-like uint8 image
whose space is tagged ``metadata["kind"] == "image"``.
"""
import os
import tempfile

import _common
from _common import bench, report, cli_args, ImageEnv, QUICK

import numpy as np

from unienv_interface.remote import RemoteClient, RemoteServer, wire
from unienv_interface.remote.compression import JpegCodec
from unienv_interface.space import DictSpace

CASES = [(None, None), (None, (84, 84, 3)), (64, (84, 84, 3))]
QUICK_CASES = [(None, None), (None, (84, 84, 3))]
TRANSPORTS = ("memory", "tcp", "unix", "websocket")


def _label(batch, img):
    return ("state" if img is None else "x".join(map(str, img))) + ("" if batch is None else f" B={batch}")


def camera_env(batch, img):
    """``ImageEnv`` with a smooth image (JPEG-friendly, unlike noise) tagged as an image."""
    env = ImageEnv(batch, img)
    if img is not None:
        h, w = img[0], img[1]
        y, x = np.mgrid[0:h, 0:w]
        smooth = np.stack([x * 255 / w, y * 255 / h, (x + y) * 64 / (h + w)], -1)
        noise = np.random.default_rng(0).normal(0, 3, env.image.shape)
        env.image[...] = np.clip(smooth + noise, 0, 255).astype(np.uint8)
        image_space = env.observation_space["image"].with_metadata(kind="image", format="rgb", layout="HWC")
        env.observation_space = DictSpace(env.backend, {**env.observation_space.spaces, "image": image_space})
    return env


def jpeg_case(batch, img):
    if img is None:
        return
    image = camera_env(batch, img).image
    codec, options = JpegCodec(), {"quality": 90, "layout": "HWC", "format": "rgb"}
    blobs = codec.encode(image, options)
    ratio = image.nbytes / sum(map(len, blobs))
    report(f"jpeg q90 encode {_label(batch, img)} ({ratio:.0f}x smaller)", bench(lambda: codec.encode(image, options)), batch)
    report(f"jpeg q90 decode {_label(batch, img)}", bench(lambda: codec.decode(blobs, image.dtype, image.shape, options)), batch)


def wire_case(batch, img):
    env = camera_env(batch, img)
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
        server.register("env", camera_env(batch, img))
        if transport == "memory":
            client = server.connect(**client_options)
        else:
            client = RemoteClient.connect(server.listen(_address(transport)), **client_options)
        with client:
            env = client.env("env")
            env.reset()
            action = np.zeros(env.action_space.shape, np.float32)
            us = bench(lambda: env.step(action), min_time=1.0, warmup=50)
    suffix = " jpeg" if client_options.get("compress") else ""
    report(f"remote[{transport}] step {_label(batch, img)}{suffix}", us, batch)
    return us


def main():
    sections = cli_args() or ["wire", "jpeg", *TRANSPORTS]
    cases = QUICK_CASES if QUICK else CASES
    for section in sections:
        for batch, img in cases:
            if section == "wire":
                wire_case(batch, img)
            elif section == "jpeg":
                jpeg_case(batch, img)
            else:
                step_case(section, batch, img)
                if img is not None:
                    step_case(section, batch, img, compress={"image": "jpeg:90"})


if __name__ == "__main__":
    main()
