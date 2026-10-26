"""Read-only access to the Win32 resources of the original PhotoDsk.exe.

The executable is never run; only its resource section is parsed so that the
reimplementation can use the original dialog layouts, menus, toolbar,
accelerators, string table, bitmaps and cursors.
"""
import struct

RT_CURSOR, RT_BITMAP, RT_ICON, RT_MENU, RT_DIALOG, RT_STRING = 1, 2, 3, 4, 5, 6
RT_ACCELERATOR, RT_GROUP_CURSOR, RT_GROUP_ICON = 9, 12, 14
RT_TOOLBAR = 241

_CONTROL_CLASSES = {0x80: "Button", 0x81: "Edit", 0x82: "Static", 0x83: "ListBox",
                    0x84: "ScrollBar", 0x85: "ComboBox"}


def _sz_or_ord(b, o):
    w, = struct.unpack_from("<H", b, o)
    if w == 0xFFFF:
        return struct.unpack_from("<H", b, o + 2)[0], o + 4
    e = o
    while struct.unpack_from("<H", b, e)[0]:
        e += 2
    return b[o:e].decode("utf-16le"), e + 2


class Control:
    __slots__ = ("id", "cls", "style", "rect", "text")

    def __init__(self, cid, cls, style, rect, text):
        self.id, self.cls, self.style, self.rect, self.text = cid, cls, style, rect, text

    @property
    def button_kind(self):
        return {0: "push", 1: "defpush", 2: "check", 3: "autocheck", 4: "radio", 5: "3state",
                6: "auto3state", 7: "group", 9: "autoradio", 11: "owner"}.get(self.style & 0xF, "push")

    def __repr__(self):
        return f"Control({self.id}, {self.cls!r}, {self.rect}, {self.text!r})"


class Dialog:
    def __init__(self, data):
        ex = struct.unpack_from("<HH", data, 0)[1] == 0xFFFF
        if ex:
            _, _, self.style, n, _, _, cx, cy = struct.unpack_from("<IIIHhhhh", data, 4)
            o = 26
        else:
            self.style, _, n, _, _, cx, cy = struct.unpack_from("<IIHhhhh", data, 0)
            o = 18
        self.size = (cx, cy)
        _, o = _sz_or_ord(data, o)     # menu
        _, o = _sz_or_ord(data, o)     # class
        self.title, o = _sz_or_ord(data, o)
        if self.style & 0x40:          # DS_SETFONT
            o += 6 if ex else 2
            _, o = _sz_or_ord(data, o)
        self.controls = []
        for _ in range(n):
            o = (o + 3) & ~3
            if ex:
                _, _, st, x, y, w, h, cid = struct.unpack_from("<IIIhhhhI", data, o)
                o += 24
            else:
                st, _, x, y, w, h, cid = struct.unpack_from("<IIhhhhH", data, o)
                o += 18
            cls, o = _sz_or_ord(data, o)
            text, o = _sz_or_ord(data, o)
            extra, = struct.unpack_from("<H", data, o)
            o += 2 + extra
            cls = _CONTROL_CLASSES.get(cls, cls) if isinstance(cls, int) else cls
            if cid in (0xFFFF, 0xFFFFFFFF):
                cid = -1
            self.controls.append(Control(cid, cls, st, (x, y, w, h), text))

    def control(self, cid):
        return next(c for c in self.controls if c.id == cid)


class PEResources:
    def __init__(self, data):
        self.data = data
        pe, = struct.unpack_from("<I", data, 0x3C)
        if data[pe:pe + 4] != b"PE\0\0":
            raise ValueError("not a PE file")
        nsec, = struct.unpack_from("<H", data, pe + 6)
        opt_size, = struct.unpack_from("<H", data, pe + 20)
        opt = pe + 24
        magic, = struct.unpack_from("<H", data, opt)
        dd = opt + (96 if magic == 0x10B else 112)
        self._rsrc_rva, _ = struct.unpack_from("<II", data, dd + 2 * 8)
        self._sections = []
        for i in range(nsec):
            s = opt + opt_size + 40 * i
            vsize, va, rsize, raw = struct.unpack_from("<IIII", data, s + 8)
            self._sections.append((va, max(vsize, rsize), raw))
        self._base = self._offset(self._rsrc_rva)
        self.tree = self._read_dir(0, 0)

    def _offset(self, rva):
        for va, size, raw in self._sections:
            if va <= rva < va + size:
                return rva - va + raw
        raise ValueError(f"RVA {rva:#x} outside sections")

    def _read_dir(self, off, depth):
        b = self.data
        p = self._base + off
        named, ids = struct.unpack_from("<HH", b, p + 12)
        out = {}
        for i in range(named + ids):
            name, target = struct.unpack_from("<II", b, p + 16 + 8 * i)
            if name & 0x80000000:
                q = self._base + (name & 0x7FFFFFFF)
                n, = struct.unpack_from("<H", b, q)
                key = b[q + 2:q + 2 + 2 * n].decode("utf-16le")
            else:
                key = name
            if target & 0x80000000:
                out[key] = self._read_dir(target & 0x7FFFFFFF, depth + 1)
            else:
                rva, size = struct.unpack_from("<II", b, self._base + target)
                out[key] = (rva, size)
        return out

    def raw(self, rtype, rid):
        langs = self.tree[rtype][rid]
        entry = langs.get(1033) or next(iter(langs.values()))
        rva, size = entry
        o = self._offset(rva)
        return self.data[o:o + size]

    def ids(self, rtype):
        return list(self.tree.get(rtype, {}))

    # ------------------------------------------------------------------ typed
    def dialog(self, rid):
        return Dialog(self.raw(RT_DIALOG, rid))

    def menu(self, rid):
        """Return a nested list: ("popup", text, items) | ("item", id, text) | ("sep",)."""
        b = self.raw(RT_MENU, rid)

        def items(o):
            out = []
            while True:
                flags, = struct.unpack_from("<H", b, o)
                o += 2
                if flags & 0x10:
                    text, o = _sz_or_ord(b, o)
                    sub, o = items(o)
                    out.append(("popup", text, sub))
                else:
                    cid, = struct.unpack_from("<H", b, o)
                    text, o = _sz_or_ord(b, o + 2)
                    out.append(("sep",) if cid == 0 and not text else ("item", cid, text))
                if flags & 0x80:
                    return out, o
        return items(4)[0]

    def strings(self):
        out = {}
        for block in self.ids(RT_STRING):
            b = self.raw(RT_STRING, block)
            o = 0
            for i in range(16):
                n, = struct.unpack_from("<H", b, o)
                o += 2
                if n:
                    out[(block - 1) * 16 + i] = b[o:o + 2 * n].decode("utf-16le")
                o += 2 * n
        return out

    def accelerators(self, rid):
        """Return [(flags, key, command)]; flags: 1 VIRTKEY, 4 SHIFT, 8 CONTROL, 0x10 ALT."""
        b = self.raw(RT_ACCELERATOR, rid)
        return [struct.unpack_from("<HHH", b, o)[:3] for o in range(0, len(b), 8)]

    def toolbar(self, rid):
        b = self.raw(RT_TOOLBAR, rid)
        _, w, h, n = struct.unpack_from("<HHHH", b, 0)
        return (w, h), list(struct.unpack_from(f"<{n}H", b, 8))

    def bitmap(self, rid):
        """Return a complete .bmp file for an RT_BITMAP resource."""
        dib = self.raw(RT_BITMAP, rid)
        hsize, = struct.unpack_from("<I", dib, 0)
        bits, = struct.unpack_from("<H", dib, 14)
        clr_used, = struct.unpack_from("<I", dib, 32) if hsize >= 36 else (0,)
        colors = clr_used or (1 << bits if bits <= 8 else 0)
        off = 14 + hsize + 4 * colors
        return b"BM" + struct.pack("<IHHI", 14 + len(dib), 0, 0, off) + dib

    def cursor(self, group_id):
        """Return (.cur file bytes, hotspot) of the first image of a cursor group."""
        g = self.raw(RT_GROUP_CURSOR, group_id)
        w, h2, _, _, _, cid = struct.unpack_from("<HHHHIH", g, 6)
        data = self.raw(RT_CURSOR, cid)
        hx, hy = struct.unpack_from("<HH", data, 0)
        dib = data[4:]
        h = h2 // 2
        cur = struct.pack("<HHH", 0, 2, 1) + struct.pack("<BBBBHHII", w & 0xFF, h & 0xFF, 0, 0, hx, hy, len(dib), 22)
        return cur + dib, (hx, hy)

    def icon(self, group_id):
        """Return an .ico file containing every image of an icon group."""
        g = self.raw(RT_GROUP_ICON, group_id)
        n, = struct.unpack_from("<H", g, 4)
        entries, images = [], []
        for i in range(n):
            w, h, cc, r, planes, bpp, size, iid = struct.unpack_from("<BBBBHHIH", g, 6 + 14 * i)
            images.append(self.raw(RT_ICON, iid))
            entries.append((w, h, cc, r, planes, bpp))
        out = struct.pack("<HHH", 0, 1, n)
        off = 6 + 16 * n
        for (w, h, cc, r, planes, bpp), img in zip(entries, images):
            out += struct.pack("<BBBBHHII", w, h, cc, r, planes, bpp, len(img), off)
            off += len(img)
        return out + b"".join(images)
