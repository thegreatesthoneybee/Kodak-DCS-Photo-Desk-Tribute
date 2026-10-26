"""InstallShield 6+ cabinet reader (format logic ported from unshield 1.6.2).

Works entirely in memory on a ``{name: bytes}`` mapping that contains
``data1.hdr`` and ``data<N>.cab`` volumes, as produced by ``mscab.extract``.
"""
import struct
import zlib

FILE_SPLIT = 1
FILE_OBFUSCATED = 2
FILE_COMPRESSED = 4
FILE_INVALID = 8
IS_SIGNATURE = 0x28635349


def _u16(b, o):
    return struct.unpack_from("<H", b, o)[0]


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def _u64(b, o):
    return struct.unpack_from("<Q", b, o)[0]


def _cstr(b, o):
    return b[o:b.index(b"\x00", o)].decode("cp1252")


class Cabinet:
    def __init__(self, volumes):
        """volumes: mapping of file name (case-insensitive) to contents."""
        self.volumes = {k.lower(): v for k, v in volumes.items()}
        self.hdr = self.volumes["data1.hdr"]
        if _u32(self.hdr, 0) != IS_SIGNATURE:
            raise ValueError("not an InstallShield cabinet header")
        version = _u32(self.hdr, 4)
        major = (version >> 12) & 0xF if version >> 24 == 1 else (version & 0xFFFF) // 100
        if major < 6:
            raise ValueError(f"InstallShield major version {major} not supported")
        self.desc = _u32(self.hdr, 12)
        d = self.desc
        self.file_table_offset = _u32(self.hdr, d + 0x0C)
        self.directory_count = _u32(self.hdr, d + 0x1C)
        self.file_count = _u32(self.hdr, d + 0x28)
        self.file_table_offset2 = _u32(self.hdr, d + 0x2C)
        base = d + self.file_table_offset
        self.table = [_u32(self.hdr, base + 4 * i) for i in range(self.directory_count + self.file_count)]

    def file_groups(self):
        """Yield (group name, first file index, last file index)."""
        for i in range(71):
            nxt = _u32(self.hdr, self.desc + 0x3E + 4 * i)
            while nxt:
                entry = self.desc + nxt
                p = self.desc + _u32(self.hdr, entry + 4)
                nxt = _u32(self.hdr, entry + 8)
                name = _cstr(self.hdr, self.desc + _u32(self.hdr, p))
                yield name, _u32(self.hdr, p + 0x16), _u32(self.hdr, p + 0x1A)

    def _ft(self, off):
        return self.desc + self.file_table_offset + off

    def descriptor(self, index):
        p = self._ft(self.file_table_offset2 + index * 0x57)
        h = self.hdr
        return {
            "flags": _u16(h, p),
            "expanded": _u64(h, p + 2),
            "compressed": _u64(h, p + 10),
            "data_offset": _u64(h, p + 18),
            "name": _cstr(h, self._ft(_u32(h, p + 0x3A))),
            "volume": _u16(h, p + 0x55),
        }

    def is_valid(self, index):
        fd = self.descriptor(index)
        return not fd["flags"] & FILE_INVALID and fd["data_offset"] != 0

    def _volume(self, n):
        return self.volumes[f"data{n}.cab"]

    def _raw(self, index, fd):
        compressed = fd["flags"] & FILE_COMPRESSED
        wanted = fd["compressed"] if compressed else fd["expanded"]
        out = bytearray()
        volume = fd["volume"]
        while len(out) < wanted:
            vol = self._volume(volume)
            if fd["flags"] & FILE_SPLIT:
                v = struct.unpack_from("<16I", vol, 20)
                if index == v[3] and v[10] != 0x7FFFFFFF:
                    off, size = v[10] | v[11] << 32, (v[14] | v[15] << 32) if compressed else (v[12] | v[13] << 32)
                elif index == v[2]:
                    off, size = v[4] | v[5] << 32, (v[8] | v[9] << 32) if compressed else (v[6] | v[7] << 32)
                else:
                    raise ValueError(f"file {index} not found in volume {volume}")
            else:
                off, size = fd["data_offset"], wanted
            chunk = vol[off:off + size]
            if len(chunk) != size:
                raise ValueError(f"short read in data{volume}.cab")
            out += chunk
            volume += 1
        if fd["flags"] & FILE_OBFUSCATED:
            for i in range(len(out)):
                x = out[i] ^ 0xD5
                x = ((x >> 2) | (x << 6)) & 0xFF
                out[i] = (x - (i % 0x47)) & 0xFF
        return bytes(out)

    def extract(self, index):
        """Return (name, bytes) of one file."""
        fd = self.descriptor(index)
        raw = self._raw(index, fd)
        if fd["flags"] & FILE_COMPRESSED:
            # Stream of [uint16 length][raw deflate block] records.
            parts = []
            pos = 0
            while pos < len(raw):
                n = _u16(raw, pos)
                block = raw[pos + 2:pos + 2 + n]
                if len(block) != n:
                    raise ValueError(f"{fd['name']}: truncated compressed block")
                d = zlib.decompressobj(-15)
                parts.append(d.decompress(block) + d.flush(zlib.Z_SYNC_FLUSH))
                pos += 2 + n
            data = b"".join(parts)
        else:
            data = raw
        if len(data) != fd["expanded"]:
            raise ValueError(f"{fd['name']}: expected {fd['expanded']} bytes, got {len(data)}")
        return fd["name"], data

    def extract_group(self, group):
        """Return {name: bytes} for all valid files of the named file group."""
        for name, first, last in self.file_groups():
            if name == group:
                return dict(self.extract(i) for i in range(first, last + 1) if self.is_valid(i))
        raise KeyError(f"file group {group!r} not found")
