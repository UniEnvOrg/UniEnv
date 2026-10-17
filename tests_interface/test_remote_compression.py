"""Opt-in JPEG compression of image leaves (server -> client only)."""

from __future__ import annotations

import json
import struct

import numpy as np
import pytest

from unienv_interface.backends.numpy import NumpyComputeBackend as B
from unienv_interface.env_base.env import Env
from unienv_interface.remote import RemoteServer, WireError, compression, wire
from unienv_interface.remote.compression import JpegCodec, parse_spec
from unienv_interface.space import BoxSpace, DictSpace
from unienv_interface.world import WorldEnv
from unienv_interface.world.node import WorldNode

from tests_common.env_worlds import SimWorld
from tests_common.remote_envs import connect


def image_space(shape, dtype=np.uint8, **metadata):
    return BoxSpace(B, 0, 255, dtype, shape=shape, metadata=metadata)


def picture(*shape, channel_axis=-1):
    """A smooth camera-like uint8 image batch (JPEG-friendly)."""
    height, width = (shape[-3], shape[-2]) if channel_axis == -1 else (shape[-2], shape[-1])
    y, x = np.mgrid[0:height, 0:width]
    noise = np.random.default_rng(0).normal(0, 3, shape)
    planes = [x * 255 / width, y * 255 / height, (x + y) * 64 / (height + width), np.full_like(x, 200.0)]
    channels = shape[channel_axis]
    base = np.stack(planes[:channels] if channels > 1 else planes[1:2], channel_axis)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def close_to(decoded, original, tolerance=4.0):
    assert decoded.shape == original.shape and decoded.dtype == np.uint8
    error = np.abs(decoded.astype(np.float32) - original).mean()
    assert error < tolerance, error
    return error


@pytest.fixture(params=["opencv", "pillow"])
def library(request, monkeypatch):
    if request.param == "pillow":
        monkeypatch.setattr(compression, "cv2", None)
    elif compression.cv2 is None:
        pytest.skip("OpenCV is not installed")
    return request.param


@pytest.mark.parametrize("shape,layout,image_format", [
    ((48, 64, 3), "HWC", "rgb"), ((48, 64, 3), "HWC", "bgr"), ((48, 64, 1), "HWC", "gray"),
    ((3, 48, 64), "CHW", "rgb"), ((2, 3, 48, 64, 3), "HWC", "rgb"),
])
def test_jpeg_codec_roundtrip(library, shape, layout, image_format):
    original = picture(*shape, channel_axis=-1 if layout == "HWC" else -3)
    codec = JpegCodec()
    options = {"quality": 90, "layout": layout, "format": image_format}
    blobs = codec.encode(original, options)
    assert len(blobs) == int(np.prod(shape[:-3])) and sum(map(len, blobs)) < original.nbytes / 2
    decoded = codec.decode([memoryview(b) for b in blobs], np.dtype(np.uint8), shape, options)
    close_to(decoded, original)
    if image_format != "gray":  # channel order is preserved, not swapped
        channels = decoded if layout == "HWC" else np.moveaxis(decoded, -3, -1)
        assert channels[..., 0].mean() > channels[..., 2].mean() + 30


def test_jpeg_codecs_interoperate(monkeypatch):
    if compression.cv2 is None:
        pytest.skip("OpenCV is not installed")
    original = picture(32, 40, 3)
    options = {"quality": 90, "layout": "HWC", "format": "rgb"}
    blobs = JpegCodec().encode(original, options)  # OpenCV
    monkeypatch.setattr(compression, "cv2", None)
    close_to(JpegCodec().decode(blobs, np.dtype(np.uint8), original.shape, options), original)  # Pillow


class CameraEnv(Env):
    """Observation with image leaves of every flavour (only ``rgb`` / ``chw`` / ``gray`` compressible)."""

    backend, device, batch_size = B, None, None
    metadata = {"render_modes": []}

    def __init__(self, batch: tuple = ()):
        self.action_space = BoxSpace(B, -1.0, 1.0, np.float32, shape=(1,))
        self.context_space = image_space(batch + (16, 16, 3), kind="image")
        self.observation_space = DictSpace(B, {
            "rgb": image_space(batch + (48, 64, 3), kind="image", format="rgb", layout="HWC"),
            "chw": image_space(batch + (3, 48, 64), kind="image", layout="CHW"),
            "gray": image_space(batch + (48, 64, 1), kind="image", format="gray"),
            "rgba": image_space(batch + (48, 64, 4), kind="image", format="rgba"),
            "depth": image_space(batch + (48, 64, 1), np.float32, kind="depth"),
            "float_image": image_space(batch + (48, 64, 3), np.float32, kind="image"),
            "plain": BoxSpace(B, 0, 255, np.uint8, shape=batch + (48, 64, 3)),  # no metadata
        })
        self.obs = {
            "rgb": picture(*batch, 48, 64, 3), "chw": picture(*batch, 3, 48, 64, channel_axis=-3),
            "gray": picture(*batch, 48, 64, 1), "rgba": picture(*batch, 48, 64, 4),
            "depth": np.random.default_rng(1).random(batch + (48, 64, 1), np.float32),
            "float_image": picture(*batch, 48, 64, 3).astype(np.float32), "plain": picture(*batch, 48, 64, 3),
        }
        self.context = picture(*batch, 16, 16, 3)

    def reset(self, *, mask=None, seed=None, **kwargs):
        return self.context, self.obs, {}

    def step(self, action):
        return self.obs, 0.0, False, False, {"raw": self.obs["rgb"]}

    def render(self):
        return self.obs["rgb"]


COMPRESSED = ("rgb", "chw", "gray")


def check_observation(observation, original, compressed):
    for key, value in original.items():
        if compressed and key in COMPRESSED:
            close_to(observation[key], value)
            assert not np.array_equal(observation[key], value)
        else:
            np.testing.assert_array_equal(observation[key], value)
            assert observation[key].dtype == value.dtype


@pytest.mark.parametrize("batch", [(), (2,)])
def test_server_compresses_only_image_leaves_on_request(transport_name, batch):
    hosted = CameraEnv(batch)
    with RemoteServer() as server:
        server.register("camera", hosted)
        with connect(server, transport_name) as raw, \
                connect(server, transport_name, compress={"image": "jpeg:90"}) as jpeg:
            sizes = {}
            for name, client in (("raw", raw), ("jpeg", jpeg)):
                recv = client.transport.recv

                def measure(recv=recv, name=name):
                    document, buffers = recv()
                    sizes[name] = sum(len(memoryview(b).cast("B")) for b in buffers)
                    return document, buffers

                client.transport.recv = measure
                client.env("camera").render()  # the reader thread picks up the wrapper after this
            assert jpeg.env("camera").observation_space["rgb"].metadata == {"kind": "image", "format": "rgb", "layout": "HWC"}
            context, observation, _ = raw.env("camera").reset()
            check_observation(observation, hosted.obs, compressed=False)
            np.testing.assert_array_equal(context, hosted.context)
            context, observation, _ = jpeg.env("camera").reset()
            check_observation(observation, hosted.obs, compressed=True)
            close_to(context, hosted.context)  # context_space is an image too
            compressible = hosted.context.nbytes + sum(hosted.obs[key].nbytes for key in COMPRESSED)
            assert sizes["raw"] - sizes["jpeg"] > 0.6 * compressible
            observation, *_, info = jpeg.env("camera").step(np.zeros(1, np.float32))
            check_observation(observation, hosted.obs, compressed=True)
            np.testing.assert_array_equal(info["raw"], hosted.obs["rgb"])  # no space: raw
            np.testing.assert_array_equal(jpeg.env("camera").render(), hosted.obs["rgb"])
            if batch:  # the client still sends raw arrays
                mask = np.array([True, False])
                jpeg.env("camera").reset(mask=mask)


@pytest.fixture(params=["memory", "tcp"])
def transport_name(request):
    return request.param


def test_subscribers_get_their_own_compression():
    hosted = CameraEnv()
    with RemoteServer() as server:
        server.register("camera", hosted)
        with server.connect() as controller, server.connect(compress={"image": "jpeg:80"}) as viewer, \
                controller.subscribe("camera") as raw_stream, viewer.subscribe("camera") as jpeg_stream:
            controller.env("camera").step(np.zeros(1, np.float32))
            check_observation(raw_stream.next(timeout=5).value[0], hosted.obs, compressed=False)
            check_observation(jpeg_stream.next(timeout=5).value[0], hosted.obs, compressed=True)


class CameraNode(WorldNode):
    name = "camera"
    after_reset_priorities = {0}

    def __init__(self, world):
        self.world = world
        self.observation_space = image_space((48, 64, 3), kind="image")
        self.image = picture(48, 64, 3)

    def get_observation(self):
        return self.image


def test_components_mode_node_observations_and_backends():
    torch = pytest.importorskip("torch")
    from unienv_interface.backends.pytorch import PyTorchComputeBackend as T
    world = SimWorld()
    node = CameraNode(world)
    with RemoteServer() as server:
        server.register_recursive("sim", WorldEnv(world, node))
        with server.connect(compress={"image": "jpeg"}, backend=T) as client:
            env = WorldEnv(client.world("sim/world"), client.node("sim/node"))
            _, observation, _ = env.reset()
            assert isinstance(observation, torch.Tensor)
            close_to(observation.numpy(), node.image)


@pytest.mark.parametrize("spec,match", [
    ("jpeg", "must be a dict"), ({"depth": "jpeg"}, "Cannot compress"), ({"image": "webp"}, "Unknown codec"),
    ({"image": "jpeg:0"}, "quality"), ({"image": "jpeg:101"}, "quality"),
])
def test_invalid_specs_are_rejected(spec, match):
    with pytest.raises(ValueError, match=match):
        parse_spec(spec)
    with RemoteServer() as server, pytest.raises(ValueError, match=match):
        server.connect(compress=spec)
    assert parse_spec({"image": "jpeg"}) == {"image": ("jpeg", {"quality": 90})}


def jpeg_frame(blobs, shape, **options):
    node = {"__t": "nd", "d": "|u1", "s": list(shape), "c": "jpeg", "o": options, "b": list(range(len(blobs)))}
    document = json.dumps(node).encode()
    return (struct.pack("<II", len(blobs), len(document)) + b"".join(struct.pack("<Q", len(b)) for b in blobs)
            + document + b"".join(blobs))


def test_malformed_jpeg_arrays():
    image = picture(16, 16, 3)
    blob = JpegCodec().encode(image, {"layout": "HWC"})[0]
    good = {"layout": "HWC", "format": "rgb"}
    np.testing.assert_array_less(np.abs(wire.loads(jpeg_frame([blob], (16, 16, 3), **good)).astype(int) - image), 40)
    cases = [
        (jpeg_frame([blob], (16, 16, 3), layout="NHWC"), "Invalid JPEG"),
        (jpeg_frame([blob], (16, 16), **good), "Invalid JPEG"),
        (jpeg_frame([blob, blob], (16, 16, 3), **good), "number of images"),
        (jpeg_frame([blob], (16, 32, 3), **good), "does not match"),
        (jpeg_frame([blob], (16, 16, 1), **good), "does not match"),
        (jpeg_frame([b"not a jpeg"], (16, 16, 3), **good), "Not a JPEG"),
        (jpeg_frame([blob[:2] + b"\x00" * 20], (16, 16, 3), **good), "Corrupt"),
        (jpeg_frame([blob[:2] + b"\xff\xe0\x00\x10" + b"\x00" * 14], (16, 16, 3), **good), "no frame header"),
    ]
    for payload, match in cases:
        with pytest.raises(WireError, match=match):
            wire.loads(payload)
