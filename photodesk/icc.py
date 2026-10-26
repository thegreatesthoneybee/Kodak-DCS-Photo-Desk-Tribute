"""ICC v2 profile reading: lut16 (mft2) evaluation, matrix/TRC working spaces, Lab/XYZ helpers."""
import struct
from pathlib import Path

import numpy as np

D50 = np.array([0.9642, 1.0, 0.8249])


class Lut16:
    """ICC lut16Type: input curves -> 3D CLUT -> output curves (matrix must be identity)."""

    def __init__(self, tag):
        if tag[:4] != b"mft2":
            raise ValueError(f"unsupported LUT type {tag[:4]!r}")
        nin, nout, grid = tag[8], tag[9], tag[10]
        if nin != 3 or nout != 3:
            raise ValueError("only 3-in/3-out LUTs are supported")
        matrix = struct.unpack_from(">9i", tag, 12)
        if matrix != (65536, 0, 0, 0, 65536, 0, 0, 0, 65536):
            raise ValueError("non-identity mft2 matrix is not supported")
        n_in, n_out = struct.unpack_from(">HH", tag, 48)
        p = 52
        self.in_curves = np.frombuffer(tag, ">u2", 3 * n_in, p).reshape(3, n_in).astype(np.float32) / 65535
        p += 6 * n_in
        self.clut = np.frombuffer(tag, ">u2", 3 * grid ** 3, p).reshape(grid, grid, grid, 3).astype(np.float32) / 65535
        p += 6 * grid ** 3
        self.out_curves = np.frombuffer(tag, ">u2", 3 * n_out, p).reshape(3, n_out).astype(np.float32) / 65535

    def __call__(self, x):
        x = np.asarray(x, np.float32)
        x = np.stack([_curve(self.in_curves[c], x[..., c]) for c in range(3)], -1)
        y = trilinear(self.clut, x)
        return np.stack([_curve(self.out_curves[c], y[..., c]) for c in range(3)], -1)


def _curve(table, x):
    return np.interp(x, np.linspace(0, 1, len(table), dtype=np.float32), table).astype(np.float32)


def trilinear(lut, x):
    """Sample an (n, n, n, 3) lattice at points x in [0, 1]^3."""
    n = lut.shape[0]
    x = np.clip(np.asarray(x, np.float32), 0, 1) * (n - 1)
    i = np.minimum(x.astype(np.int32), n - 2)
    f = x - i
    i0, i1, i2 = i[..., 0], i[..., 1], i[..., 2]
    f0, f1, f2 = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    c00 = lut[i0, i1, i2] * (1 - f0) + lut[i0 + 1, i1, i2] * f0
    c01 = lut[i0, i1, i2 + 1] * (1 - f0) + lut[i0 + 1, i1, i2 + 1] * f0
    c10 = lut[i0, i1 + 1, i2] * (1 - f0) + lut[i0 + 1, i1 + 1, i2] * f0
    c11 = lut[i0, i1 + 1, i2 + 1] * (1 - f0) + lut[i0 + 1, i1 + 1, i2 + 1] * f0
    c0 = c00 * (1 - f1) + c10 * f1
    c1 = c01 * (1 - f1) + c11 * f1
    return c0 * (1 - f2) + c1 * f2


class Profile:
    def __init__(self, path):
        self.path = Path(path)
        self.data = self.path.read_bytes()
        data = self.data
        if len(data) < 132 or data[36:40] != b"acsp":
            raise ValueError(f"{path} is not an ICC profile")
        self.device_class = data[12:16].decode("latin1")
        self.color_space = data[16:20].decode("latin1").strip()
        self.pcs = data[20:24].decode("latin1").strip()
        count, = struct.unpack_from(">I", data, 128)
        self.tags = {}
        for k in range(count):
            sig, off, size = struct.unpack_from(">4sII", data, 132 + 12 * k)
            self.tags[sig.decode("latin1")] = data[off:off + size]
        self.description = _desc(self.tags.get("desc", b"")) or self.path.stem

    def lut(self, sig="A2B0"):
        return Lut16(self.tags[sig])

    @property
    def is_working_space(self):
        """True for RGB matrix/TRC display profiles usable as an RGB working space."""
        return (self.device_class == "mntr" and self.color_space == "RGB" and self.pcs == "XYZ"
                and all(t in self.tags for t in ("rXYZ", "gXYZ", "bXYZ", "rTRC", "gTRC", "bTRC")))

    def rgb_to_xyz_matrix(self):
        """Columns are the D50 XYZ of the R, G, B primaries."""
        cols = [struct.unpack_from(">3i", self.tags[t], 8) for t in ("rXYZ", "gXYZ", "bXYZ")]
        return np.array(cols, np.float64).T / 65536

    def trc(self, channel):
        """Return increasing (encoded, linear) samples of a channel's tone curve."""
        tag = self.tags[channel + "TRC"]
        x = np.linspace(0, 1, 4096)
        if tag[:4] == b"curv":
            n, = struct.unpack_from(">I", tag, 8)
            if n == 0:
                return x, x
            if n == 1:
                return x, x ** (struct.unpack_from(">H", tag, 12)[0] / 256)
            y = np.frombuffer(tag, ">u2", n, 12).astype(np.float64) / 65535
            return np.linspace(0, 1, n), y
        if tag[:4] == b"para":
            func, = struct.unpack_from(">H", tag, 8)
            count = {0: 1, 1: 3, 2: 4, 3: 5, 4: 7}[func]
            p = [v / 65536 for v in struct.unpack_from(f">{count}i", tag, 12)] + [0.0] * 6
            g, a, b, c, d, e, f = p[:7]
            if func == 0:
                y = x ** g
            elif func == 1:
                y = np.where(x >= -b / a, np.maximum(a * x + b, 0) ** g, 0)
            elif func == 2:
                y = np.where(x >= -b / a, np.maximum(a * x + b, 0) ** g + c, c)
            elif func == 3:
                y = np.where(x >= d, np.maximum(a * x + b, 0) ** g, c * x)
            else:
                y = np.where(x >= d, np.maximum(a * x + b, 0) ** g + e, c * x + f)
            return x, y
        raise ValueError(f"unsupported TRC type {tag[:4]!r}")

    def encoder(self):
        """Return f(XYZ D50) -> encoded RGB in [0, 1] for a working-space profile."""
        inv = np.linalg.inv(self.rgb_to_xyz_matrix()).astype(np.float32)
        curves = [self.trc(c) for c in "rgb"]

        def encode(xyz):
            lin = np.clip(np.asarray(xyz, np.float32) @ inv.T, 0, 1)
            return np.stack([np.interp(lin[..., i], y, x) for i, (x, y) in enumerate(curves)],
                            -1).astype(np.float32)
        return encode


def _desc(tag):
    if tag[:4] == b"desc":
        n, = struct.unpack_from(">I", tag, 8)
        return tag[12:12 + n].split(b"\x00")[0].decode("latin1").strip()
    if tag[:4] == b"mluc":
        n, size = struct.unpack_from(">II", tag, 8)
        if n:
            length, off = struct.unpack_from(">II", tag, 16 + 4)
            return tag[off:off + length].decode("utf-16-be").strip("\x00 ")
    return ""


# ICC v2 16-bit PCS encodings (values normalized to [0, 1] of 65535).
def lab_decode(v):
    v = np.asarray(v, np.float32) * 65535
    return np.stack([v[..., 0] * (100 / 65280), v[..., 1] / 256 - 128, v[..., 2] / 256 - 128], -1)


def lab_encode(lab):
    lab = np.asarray(lab, np.float32)
    return np.stack([lab[..., 0] * (65280 / 100), (lab[..., 1] + 128) * 256, (lab[..., 2] + 128) * 256], -1) / 65535


def xyz_decode(v):
    return np.asarray(v, np.float32) * (65535 / 32768)


def xyz_encode(xyz):
    return np.asarray(xyz, np.float32) * (32768 / 65535)


def lab_to_xyz(lab):
    fy = (lab[..., 0] + 16) / 116
    fx = fy + lab[..., 1] / 500
    fz = fy - lab[..., 2] / 200
    d = 6 / 29

    def finv(t):
        return np.where(t > d, t ** 3, 3 * d * d * (t - 4 / 29))

    return np.stack([finv(fx), finv(fy), finv(fz)], -1).astype(np.float32) * D50.astype(np.float32)


def xyz_to_lab(xyz):
    t = np.asarray(xyz, np.float32) / D50.astype(np.float32)
    d = 6 / 29
    f = np.where(t > d ** 3, np.cbrt(np.maximum(t, 0)), t / (3 * d * d) + 4 / 29)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]), 200 * (f[..., 1] - f[..., 2])], -1)
