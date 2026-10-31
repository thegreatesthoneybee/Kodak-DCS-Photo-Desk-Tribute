"""Contact sheet document: image entries, their settings, selection-wide edits and undo."""
import dataclasses
import datetime
import os
import shutil
from pathlib import Path

from .. import camera, pipeline

RAW_SUFFIXES = {".dcr", ".kdc", ".dc2", ".k25"}
FINISHED_SUFFIXES = {".tif", ".tiff", ".jpg", ".jpeg"}

# Undo/Redo menu texts (string table ids) of each kind of edit.
UNDO = {
    "exposure": (62077, 62078), "exposure_tone": (62079, 62080), "color_balance": (62081, 62082),
    "white_point": (62083, 62084), "annotation": (62085, 62086), "sharpening": (62087, 62088),
    "noise": (62089, 62090), "click_balance": (62093, 62094), "balance": (62095, 62096),
    "lighting": (62097, 62098), "look": (62099, 62100), "auto_scene": (62101, 62102),
    "rotate": (62103, 62104), "crop": (62105, 62106), "original": (62107, 62108),
    "tag": (62109, 62110), "untag": (62111, 62112), "iptc": (62113, 62114), "rename": (62115, 62116),
    "paste_special": (62124, 62125), "lens": (62153, 62154), "auto_exposure": (62077, 62078),
}
CANT_UNDO, CANT_REDO = 62091, 62092

# Paste Special groups (check box id in dialog 292 -> Settings fields).
PASTE_GROUPS = {
    1338: ("crop",), 1340: ("rotation",), 1341: ("exposure",), 1342: ("tone", "tone_enabled"),
    1344: ("balance", "click_wb", "auto_balance_wb", "lighting", "white_point", "white_point_ref", "kelvin_mode", "kelvin_illumination", "saved_balance_point", "color_balance"),
    1352: ("lens_optimization",), 1346: ("look",),
    1347: ("noise_type", "noise_radius", "noise_strength", "noise_luma", "moire_threshold"),
    1348: ("sharpen_amount", "sharpen_radius", "sharpen_threshold"), 1350: ("iptc",), 1351: ("tagged",),
}
ALL_FIELDS = tuple(f for g in PASTE_GROUPS.values() for f in g)
FINISHED_FIELDS = ("crop", "rotation", "iptc", "tagged")
CLICK_BALANCE_FIELDS = frozenset((
    "balance", "click_wb", "auto_balance_wb", "lighting", "white_point", "white_point_ref",
    "kelvin_mode", "kelvin_illumination", "saved_balance_point",
))


@dataclasses.dataclass
class Entry:
    path: Path
    settings: pipeline.Settings
    make: str = ""
    model: str = ""
    thumb: object = None          # QImage once rendered
    thumb_gen: int = 0            # bumped per thumbnail re-render request; older results are dropped
    raw: object = None            # decoded RawImage (half size) while an image window is open
    small: object = None          # reduced RawImage used for thumbnails
    meta: dict = None             # camera metadata once decoded
    dirty: bool = False           # settings changed since last Save

    _raw_kind: bool = None
    _serial: str = None
    settings_from_sidecar: bool = False

    @property
    def serial(self):
        if self._serial is None:
            self._serial = camera.read_serial(self.path)
        return self._serial

    @property
    def is_raw(self):
        """DCS raw (all processing options) as opposed to a finished TIFF/JPEG file."""
        if self._raw_kind is None:
            self._raw_kind = is_raw_file(self.path)
        return self._raw_kind

    @property
    def caps(self):
        return camera.capabilities(self.model)

    @property
    def read_only(self):
        return not os.access(self.path, os.W_OK)

    @property
    def sounds(self):
        return sorted(self.path.parent.glob(self.path.stem + "*.wav")) + sorted(
            self.path.parent.glob(self.path.stem + "*.WAV"))

    @property
    def date(self):
        return datetime.datetime.fromtimestamp(self.path.stat().st_mtime)

    @property
    def file_type(self):
        """'R' raw, 'T' TIFF, 'J' JPEG, 'E' ERI JPEG (thumbnail info icon)."""
        s = self.path.suffix.lower()
        if self.is_raw:
            return "R"
        return "T" if s in (".tif", ".tiff") else "J"


def is_raw_file(path):
    s = Path(path).suffix.lower()
    if s in RAW_SUFFIXES:
        return True
    if s not in (".tif", ".tiff"):
        return False
    import rawpy
    try:
        with rawpy.imread(str(path)):
            return True
    except Exception:  # LibRaw rejects finished (non-raw) TIFF files
        return False


def is_image(p):
    return p.is_file() and p.suffix.lower() in RAW_SUFFIXES | FINISHED_SUFFIXES


def paste_into(entry, settings, source, fields, source_serial):
    for f in fields:
        if not entry.is_raw and f not in FINISHED_FIELDS:
            continue
        if (f in CLICK_BALANCE_FIELDS and source.balance == "click" and
                (not source_serial or entry.serial != source_serial)):
            continue
        setattr(settings, f, getattr(source, f))


class Sheet:
    """A folder opened as a contact sheet."""

    def __init__(self, folder, files=None):
        """files: only these images (an image opened by itself with File > Open, no contact sheet)."""
        self.folder = Path(folder)
        self.files = {Path(f) for f in files} if files is not None else None
        self.entries = []
        for p in sorted(self._present()):
            make, model = camera.read_make_model(p)
            self.entries.append(Entry(p, pipeline.load_settings(p), make, model,
                                      settings_from_sidecar=pipeline.has_saved_settings(p)))
        self.undo_stack, self.redo_stack = [], []
        self.sort = "name"

    def _present(self):
        if self.files is not None:
            return {f for f in self.files if f.is_file()}
        return {f for f in self.folder.iterdir() if is_image(f)}

    def sort_by(self, key):
        self.sort = key
        if key == "date":
            self.entries.sort(key=lambda e: (e.date, e.path.name.lower()))
        else:
            self.entries.sort(key=lambda e: e.path.name.lower())

    def find(self, path):
        return next((e for e in self.entries if e.path == Path(path)), None)

    # ------------------------------------------------------------ edits
    def apply(self, entries, kind, change):
        """change(entry, settings copy) mutates the copy; recorded as one undoable edit."""
        before, after = {}, {}
        for e in entries:
            s = e.settings.copy()
            change(e, s)
            if s.to_json() != e.settings.to_json():
                before[id(e)], after[id(e)] = (e, e.settings), (e, s)
                e.settings, e.dirty = s, True
        if after:
            self._push(kind, lambda: self._restore(before), lambda: self._restore(after))
        return [e for e, _ in after.values()]

    @staticmethod
    def _restore(states):
        for e, s in states.values():
            e.settings, e.dirty = s, True
        return [e for e, _ in states.values()]

    def _push(self, kind, undo, redo):
        self.undo_stack.append((kind, undo, redo))
        self.redo_stack.clear()

    def set_fields(self, entries, kind, **fields):
        return self.apply(entries, kind, lambda e, s: [setattr(s, k, v) for k, v in fields.items()])

    def paste(self, entries, source, fields, kind="paste_special", source_serial=None):
        """Paste settings copied from an image whose camera serial number is source_serial.

        Finished files only take the settings they support (crop, rotation, IPTC, tag).
        Click balance is only pasted onto images from the same camera (same serial number)."""
        return self.apply(entries, kind, lambda e, s: paste_into(e, s, source, fields, source_serial))

    def undo(self):
        """Undo the last edit; returns the affected entries."""
        if not self.undo_stack:
            return []
        item = self.undo_stack.pop()
        self.redo_stack.append(item)
        return item[1]()

    def redo(self):
        if not self.redo_stack:
            return []
        item = self.redo_stack.pop()
        self.undo_stack.append(item)
        return item[2]()

    def undo_text_id(self):
        return UNDO[self.undo_stack[-1][0]][0] if self.undo_stack else CANT_UNDO

    def redo_text_id(self):
        return UNDO[self.redo_stack[-1][0]][1] if self.redo_stack else CANT_REDO

    # ------------------------------------------------------------ files
    def save(self, entries):
        """Save settings of DCS raw images. Changes to finished files are never saved this way (only
        Save As to Standard TIFF/JPEG keeps them), and read-only images cannot be saved in place;
        returns the read-only DCS entries that still need Save As."""
        need_save_as = []
        for e in entries:
            if not e.is_raw:
                continue
            if e.read_only:
                need_save_as.append(e)
                continue
            pipeline.save_settings(e.path, e.settings)
            e.settings_from_sidecar = True
            e.dirty = False
        return need_save_as

    def dirty(self):
        return [e for e in self.entries if e.dirty]

    def rename(self, entries, names, on_error=None, record=True):
        """Rename entries to the file names given; moves sidecars and sounds too.

        A file whose new name already exists ("exists"), or that cannot be renamed ("denied" or
        "failed"), is skipped; on_error(entry, kind) returns False to stop renaming the rest.
        Returns the entries actually renamed.
        """
        done, old_names, new_names = [], [], []
        for e, name in zip(entries, names):
            old = e.path
            dest = old.with_name(name)
            if dest == old:
                continue
            kind = None
            if dest.exists() and dest.name.lower() != old.name.lower():
                kind = "exists"
            else:
                try:
                    old.rename(dest)
                except PermissionError:
                    kind = "denied"
                except OSError:
                    kind = "failed"
            if kind:
                if on_error is not None and not on_error(e, kind):
                    break
                continue
            for snd in sorted(old.parent.glob(old.stem + "*.wav")) + sorted(old.parent.glob(old.stem + "*.WAV")):
                snd.rename(snd.with_name(dest.stem + snd.name[len(old.stem):]))
            sc = pipeline.sidecar_path(old)
            if sc.is_file():
                sc.rename(pipeline.sidecar_path(dest))
            e.path = dest
            done.append(e)
            old_names.append(old.name)
            new_names.append(dest.name)
        self.sort_by(self.sort)
        if record and done:
            self._push("rename", lambda: self.rename(done, old_names, record=False) or done,
                       lambda: self.rename(done, new_names, record=False) or done)
        return done

    def delete(self, entries, trash):
        """trash(path) moves a file to the recycle bin; returns entries actually removed."""
        gone = []
        for e in entries:
            files = [e.path] + e.sounds + ([pipeline.sidecar_path(e.path)]
                                           if pipeline.sidecar_path(e.path).is_file() else [])
            if all(trash(f) for f in files):
                gone.append(e)
        self.entries = [e for e in self.entries if e not in gone]
        return gone

    def transfer(self, paths, move, replace):
        """Copy or move image files (with their sounds and settings) into this folder.

        replace(dest) decides whether an existing file is overwritten. Returns the new paths.
        """
        done = []
        for src in map(Path, paths):
            dest = self.folder / src.name
            if src.parent.resolve() == self.folder.resolve():
                continue
            if dest.exists() and not replace(dest):
                continue
            sounds = sorted(src.parent.glob(src.stem + "*.wav")) + sorted(src.parent.glob(src.stem + "*.WAV"))
            sc = pipeline.sidecar_path(src)
            for f in [src] + sounds + ([sc] if sc.is_file() else []):
                target = self.folder / f.name
                if move:
                    shutil.move(str(f), str(target))
                else:
                    shutil.copy2(f, target)
            done.append(dest)
        self.refresh()
        return done

    def refresh(self):
        """Pick up files added to or removed from the folder."""
        known = {e.path for e in self.entries}
        present = self._present()
        self.entries = [e for e in self.entries if e.path in present]
        for p in sorted(present - known):
            make, model = camera.read_make_model(p)
            self.entries.append(Entry(p, pipeline.load_settings(p), make, model,
                                      settings_from_sidecar=pipeline.has_saved_settings(p)))
        self.sort_by(self.sort)
