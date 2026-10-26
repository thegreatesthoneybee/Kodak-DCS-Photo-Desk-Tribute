"""Microsoft Cabinet (MSCF) reader supporting stored and MSZIP folders.

Used to unpack the self-extracting Photo Desk installer without running it,
so profile installation works identically on Windows and macOS.
"""
import struct
import zlib

COMPRESS_NONE = 0
COMPRESS_MSZIP = 1


def find_cabinets(blob):
    """Yield (offset, bytes) for every well-formed cabinet embedded in blob."""
    pos = 0
    while True:
        pos = blob.find(b"MSCF\x00\x00\x00\x00", pos)
        if pos < 0:
            return
        size = struct.unpack_from("<I", blob, pos + 8)[0]
        if 36 <= size <= len(blob) - pos:
            yield pos, blob[pos:pos + size]
        pos += 4


def extract(cab):
    """Return {name: bytes} for all files in a single-volume cabinet."""
    coff_files, = struct.unpack_from("<I", cab, 16)
    n_folders, n_files, flags = struct.unpack_from("<HHH", cab, 26)
    pos = 36
    folder_reserve = data_reserve = 0
    if flags & 4:
        header_reserve, folder_reserve, data_reserve = struct.unpack_from("<HBB", cab, 36)
        pos += 4 + header_reserve
    if flags & 3:
        raise ValueError("multi-volume cabinets are not supported")

    folders = []
    for _ in range(n_folders):
        start, count, ctype = struct.unpack_from("<IHH", cab, pos)
        pos += 8 + folder_reserve
        folders.append(_read_folder(cab, start, count, ctype & 0x0F, data_reserve))

    files = {}
    pos = coff_files
    for _ in range(n_files):
        size, offset, ifolder = struct.unpack_from("<IIH", cab, pos)
        end = cab.index(b"\x00", pos + 16)
        name = cab[pos + 16:end].decode("cp1252")
        pos = end + 1
        files[name] = folders[ifolder][offset:offset + size]
    return files


def _read_folder(cab, pos, count, ctype, data_reserve):
    out = bytearray()
    for _ in range(count):
        csize, usize = struct.unpack_from("<HH", cab, pos + 4)
        pos += 8 + data_reserve
        block = cab[pos:pos + csize]
        pos += csize
        if ctype == COMPRESS_NONE:
            out += block
        elif ctype == COMPRESS_MSZIP:
            if block[:2] != b"CK":
                raise ValueError("bad MSZIP block signature")
            # Each MSZIP block may reference the previous 32 KiB of output.
            d = zlib.decompressobj(-15, zdict=bytes(out[-32768:]))
            chunk = d.decompress(block[2:]) + d.flush()
            if len(chunk) != usize:
                raise ValueError("MSZIP block size mismatch")
            out += chunk
        else:
            raise ValueError(f"unsupported cabinet compression type {ctype}")
    return bytes(out)
