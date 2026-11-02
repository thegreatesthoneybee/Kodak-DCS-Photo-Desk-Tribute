"""Kodak DCS Pro SLR demosaic, read statically from DCSPro4SLR.dll.

The DLL renders CFA images line by line through ipCopyCfaToRgb (0x65ff98a0),
ipMissingGreenAH2 (0x65ffd810), ipSmoothGreen (0x65ffcf90), ipSmoothRedBlue (0x65ffd330),
ipHorizontalChroma (0x65ff9660) and ipVerticalChroma (0x65ffc980). All stages are integer
kernels on balanced 12-bit codes; this is a whole-image transcription of them.
"""
import numpy as np

# Scene word +0x540, copied to the AH2 context +0x38/+0x3a. 0x65ffa880 sets it to 160 for
# the Pro SLR/n camera ids when the file leaves it unset (240 for other ids).
AH2_THRESHOLD = 160


def _clip12(values):
    return np.clip(values, 0, 4095)


def balance_cfa(cfa, gains, saturation):
    """ipApplyBalanceGainsCFA tables (0x65ff9d90): min((code * round(gain * 1024) + 512) >> 10, sat).

    ``gains`` is (R, G, B); the CFA must start with G R on even rows and B G on odd rows.
    """
    fixed = np.round(np.asarray(gains, np.float64) * 1024).astype(np.int64)
    out = np.empty(cfa.shape, np.int32)
    codes = cfa.astype(np.int64)
    for dy, dx, channel in ((0, 0, 1), (0, 1, 0), (1, 0, 2), (1, 1, 1)):
        out[dy::2, dx::2] = np.minimum((codes[dy::2, dx::2] * fixed[channel] + 512) >> 10, saturation)
    return out


OMEN_TILE = 16                     # ipApplyOmenCFA gain tiles are 16 x 16 sensor pixels
OMEN_TILES = (189, 283)            # 0xd0ef words; grid nodes sit on every fourth tile


def omen_tiles(grid, unit, scale, strength, baseline):
    """Q12 blue gain per 16-pixel tile built by the lens optimization object (0x65fe66d0).

    Node gain: _ftol((x * A + 4096) * 4096 / (x * B + 4096)) clamped to 0..0x3fff, where
    x = grid * (4096 // unit) - 4096, A = lo + (hi - lo) * strength / 100 for the requested
    strength (object +0x70) and B the same for the in-camera strength of tag 0xa5e (+0x5c);
    (lo, hi) is tag 0xa67. Equal strengths give unity gains. Nodes are filled in between with the
    native (a + b + 1) >> 1 midpoint chain, rows first, and the last three columns repeat col 280.
    """
    lo, hi = scale
    a = lo + (hi - lo) * (strength * 0.01)
    b = lo + (hi - lo) * (baseline * 0.01)
    x = (np.asarray(grid, np.int64) * (0x1000 // unit) - 0x1000).astype(np.float64)
    nodes = np.trunc((x * a + 4096) * 4096 / (x * b + 4096))
    nodes = np.clip(np.nan_to_num(nodes, nan=0x3fff), 0, 0x3fff).astype(np.int64)
    t = np.zeros(OMEN_TILES, np.int64)
    rows, cols = nodes.shape
    t[0:rows * 4:4, 0:cols * 4:4] = nodes

    def fill(a_, b_):
        m = (a_ + b_ + 1) >> 1
        return (a_ + m + 1) >> 1, m, (m + b_ + 1) >> 1

    for c in range(0, (cols - 1) * 4, 4):
        t[0::4, c + 1], t[0::4, c + 2], t[0::4, c + 3] = fill(t[0::4, c], t[0::4, c + 4])
    t[0::4, (cols - 1) * 4 + 1:] = t[0::4, (cols - 1) * 4:(cols - 1) * 4 + 1]
    for r in range(0, (rows - 1) * 4 - 3, 4):
        t[r + 1], t[r + 2], t[r + 3] = fill(t[r], t[r + 4])
    return t


def omen_linear(codes, gain, linearization):
    """Linear sensor codes after the Omen ``gain`` (Q12 / 4096) on B samples given as linear codes.

    ipApplyOmenCFA scales the stored CFA codes, which the camera's linearization table (tag 0x90d,
    stored -> linear) maps to linear codes only afterwards in the balance tables; the gain acts
    on the inverted (stored) code, interpolated through the table both ways. Without a table the
    stored codes are linear and the gain applies directly.
    """
    codes = np.asarray(codes, np.float64)
    if linearization is None:
        return codes * gain
    table = np.asarray(linearization, np.float64)
    index = np.arange(len(table), dtype=np.float64)
    stored = np.interp(codes, table, index) * gain
    return np.interp(np.minimum(stored, 4095), index, table)


def apply_omen_cfa(cfa, tiles):
    """ipApplyOmenCFA (0x65ffe530): B sites (odd rows, even columns) times their tile gain,
    (code * gain + 0x800) >> 12 clamped to 0..4095."""
    out = np.array(cfa, np.int64)
    rows = np.minimum(np.arange(1, out.shape[0], 2) // OMEN_TILE, OMEN_TILES[0] - 1)
    cols = np.minimum(np.arange(0, out.shape[1], 2) // OMEN_TILE, OMEN_TILES[1] - 1)
    out[1::2, 0::2] = np.clip((out[1::2, 0::2] * tiles[rows[:, None], cols[None, :]] + 0x800) >> 12, 0, 4095)
    return out


def ah2(cfa, threshold=AH2_THRESHOLD, band=512):
    """Demosaic balanced 12-bit codes (G R / B G) to a uint16 HxWx3 RGB image, in row bands."""
    height = cfa.shape[0]
    out = np.empty(cfa.shape + (3,), np.uint16)
    margin = 8   # the kernels reach five rows; an even margin keeps the CFA phase
    for start in range(0, height, band):
        stop = min(height, start + band)
        lo, hi = max(0, start - margin), min(height, stop + margin)
        out[start:stop] = _ah2(cfa[lo:hi], threshold)[start - lo:stop - lo]
    return out


def _ah2(cfa, threshold):
    height, width = cfa.shape
    if height % 2 or width % 2:
        raise ValueError("Kodak AH2 demosaic needs even image dimensions")
    pad = 2
    # Reflection about the edge sample keeps the CFA phase of every padded sample.
    c = np.pad(cfa.astype(np.int32), pad, mode="reflect")
    H, W = c.shape

    def at(a, dy, dx):
        return a[pad + dy:H - pad + dy, pad + dx:W - pad + dx]

    yy = np.arange(height)[:, None] & 1
    xx = np.arange(width)[None, :] & 1
    green = yy == xx

    # ipMissingGreenAH2 at R and B sites (X is the site colour).
    x = at(c, 0, 0)
    gl, gr, gu, gd = at(c, 0, -1), at(c, 0, 1), at(c, -1, 0), at(c, 1, 0)
    xl, xr, xu, xd = at(c, 0, -2), at(c, 0, 2), at(c, -2, 0), at(c, 2, 0)
    lap_h = 2 * x - xl - xr
    lap_v = 2 * x - xu - xd
    grad_h = np.abs(lap_h) + 2 * np.abs(gl - gr)
    grad_v = np.abs(lap_v) + 2 * np.abs(gu - gd)
    flat = (grad_h <= threshold) & (grad_v <= threshold)
    g_flat = ((gu + gd + gl + gr + 2) >> 2) + ((3 * (4 * x - xl - xr - xu - xd)) >> 5)

    def directional(g_diff, x_diff, lap, g_sum):
        gentle = (2 * np.abs(g_diff) < np.abs(lap)) & (np.abs(lap) < 2 * np.abs(x_diff))
        return (g_sum >> 1) + np.where(gentle, (3 * lap) >> 5, (3 * lap) >> 4)

    g_dir = np.where(grad_h <= grad_v, directional(gl - gr, xl - xr, lap_h, gl + gr),
                     directional(gu - gd, xu - xd, lap_v, gu + gd))
    g = np.where(green, x, _clip12(np.where(flat, g_flat, g_dir)))
    flagged = ~green & flat

    # ipSmoothGreen: a green site with at least two flagged neighbours takes their rounded mean.
    gp = np.pad(g, pad, mode="reflect")
    fp = np.pad(flagged, pad, mode="reflect").astype(np.int8)
    count = at(fp, 0, -1) + at(fp, 0, 1) + at(fp, -1, 0) + at(fp, 1, 0)
    mean4 = (at(gp, 0, -1) + at(gp, 0, 1) + at(gp, -1, 0) + at(gp, 1, 0) + 2) >> 2
    g = np.where(green & (count >= 2), _clip12(mean4), g)

    # ipSmoothRedBlue: flagged R/B sites become (21 X + 4 same-colour neighbours) / 25.
    smooth = ((21 * x + xl + xr + xu + xd) * 0x51F + 0x4000) >> 15
    x = np.where(flagged, smooth, x)

    # ipHorizontalChroma: X - G at the site, averaged across the row between sites.
    chroma = np.where(green, 0, x - g)
    cp = np.pad(chroma, pad, mode="reflect")
    between = (at(cp, 0, -1) + at(cp, 0, 1) + 1) >> 1
    row_chroma = np.where(green, between, chroma)

    # ipVerticalChroma: rows without a colour take it from the rows above and below, choosing
    # the less different diagonal pair off the site columns.
    rp = np.pad(row_chroma, pad, mode="reflect")
    up, down = at(rp, -1, 0), at(rp, 1, 0)
    ul, ur, dl, dr = at(rp, -1, -1), at(rp, -1, 1), at(rp, 1, -1), at(rp, 1, 1)
    vertical = (up + down + 1) >> 1
    d1, d2 = np.abs(ur - dl), np.abs(ul - dr)
    diagonal = np.where(d1 <= d2, (dl + ur + 1) >> 1, np.where(d2 <= d1, (dr + ul + 1) >> 1, vertical))
    red_rows = yy == 0
    cr = np.where(red_rows, row_chroma, np.where(xx == 1, vertical, diagonal))
    cb = np.where(~red_rows, row_chroma, np.where(xx == 0, vertical, diagonal))
    return np.stack([_clip12(g + cr), g, _clip12(g + cb)], -1)
