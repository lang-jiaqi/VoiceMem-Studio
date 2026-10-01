"""Versioned browser resources for the authenticated mobile pet page."""
from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from html import escape
from pathlib import Path

from starlette.datastructures import Headers
from starlette.exceptions import HTTPException
from starlette.responses import Response
from starlette.staticfiles import StaticFiles

CACHE_CONTROL = "private, max-age=31536000, immutable"
PET_SCRIPTS = (
    "avatar-parameter-controller.js", "avatar-behavior-controller.js",
    "audio-lip-sync.js", "live2d-renderer.js", "avatar-controller.js",
)
PET_VENDORS = {
    "pixi.min.js": "pixi.js/dist/browser/pixi.min.js",
    "cubism4.min.js": "pixi-live2d-display/dist/cubism4.min.js",
}
MOBILE_FILES = ("pet-mobile.css", "studio-client.js", "pet-mobile.js")
CORE_PATH = "assets/live2d/vendor/live2dcubismcore.min.js"
CORE_CDN = "https://cubism.live2d.com/sdk-web/cubismcore/live2dcubismcore.min.js"
MODEL_PATH = "assets/live2d/rattan/rattan.model3.json"


def accepts_gzip(headers: Headers) -> bool:
    """Honor an explicit gzip quality value before the wildcard fallback."""
    qualities = {}
    for item in headers.get("accept-encoding", "").lower().split(","):
        coding, *parameters = item.strip().split(";")
        quality = 1.0
        for parameter in parameters:
            if parameter.strip().startswith("q="):
                try:
                    quality = float(parameter.strip()[2:])
                except ValueError:
                    quality = 0.0
        qualities[coding] = quality
    return 0 < qualities.get("gzip", qualities.get("*", 0)) <= 1


def cached_response(data: bytes, headers: dict, scope) -> Response:
    """Use the selected representation's validator for GET and HEAD."""
    request = Headers(scope=scope)
    etag = headers["ETag"]
    tags = request.get("if-none-match", "").split(",")
    if any(tag.strip().removeprefix("W/") in {etag, "*"} for tag in tags):
        return Response(status_code=304, headers={key: value for key, value in headers.items()
                                                 if key.lower() != "content-length"})
    headers = {**headers, "Content-Length": str(len(data))}
    return Response(b"" if scope["method"] == "HEAD" else data, headers=headers)


@dataclass(frozen=True)
class GzipAsset:
    """A lossless representation prepared once, before accepting clients."""

    data: bytes
    etag: str

    @classmethod
    def prepare(cls, data: bytes):
        if len(data) < 1024:
            return None
        compressed = gzip.compress(data, compresslevel=6, mtime=0)
        if len(compressed) >= len(data) * .9:
            return None
        return cls(compressed, '"' + hashlib.sha256(compressed).hexdigest() + '"')

    def response(self, headers: dict, scope) -> Response:
        headers = {**headers, "Content-Encoding": "gzip", "Vary": "Accept-Encoding",
                   "ETag": self.etag}
        return cached_response(self.data, headers, scope)


class CachedStaticFiles(StaticFiles):
    """Cache versioned files, preserving conditional requests and path checks."""

    def __init__(self, directory, *, files=None, check_dir=True, gzip_assets=None):
        super().__init__(directory=directory, check_dir=check_dir)
        self.files = files
        self.gzip_assets = gzip_assets or {}

    async def get_response(self, path, scope):
        if self.files is not None:
            if path not in self.files:
                raise HTTPException(404)
            path = self.files[path]
        compressed = self.gzip_assets.get(path)
        request = Headers(scope=scope)
        use_gzip = compressed and accepts_gzip(request) and "range" not in request
        file_scope = scope
        if use_gzip:
            # Validate the file through StaticFiles, using the gzip ETag below.
            file_scope = {**scope, "headers": [(key, value) for key, value in scope["headers"]
                                               if key not in {b"if-none-match", b"if-modified-since"}]}
        response = await super().get_response(path, file_scope)
        if response.status_code in {200, 304}:
            response.headers["Cache-Control"] = CACHE_CONTROL
        if compressed:
            response.headers["Vary"] = "Accept-Encoding"
        if use_gzip and response.status_code == 200:
            headers = {"Cache-Control": CACHE_CONTROL,
                       "Content-Type": response.headers["content-type"]}
            return compressed.response(headers, scope)
        return response


class MobilePetAssets:
    """Compute one content version at startup; deployments restart to update it."""

    def __init__(self, pet_root: Path, ui_root: Path, vendor_root: Path):
        sources = {
            **{f"pet/{name}": pet_root / name for name in PET_SCRIPTS},
            **{f"ui/{name}": ui_root / name for name in (*MOBILE_FILES, "pet-mobile.html")},
            **{f"vendor/{name}": vendor_root / path for name, path in PET_VENDORS.items()},
        }
        sources.update({f"assets/{path.relative_to(pet_root / 'assets').as_posix()}": path
                        for path in (pet_root / "assets").rglob("*") if path.is_file()})
        digest = hashlib.sha256()
        for name, path in sorted(sources.items()):
            digest.update(name.encode() + b"\0")
            if path.is_file():
                file_digest = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                        file_digest.update(chunk)
                digest.update(file_digest.digest())
            else:
                digest.update(b"missing")
        self.version = digest.hexdigest()[:16]
        self.prefix = f"/pet/v/{self.version}"
        self.html = (ui_root / "pet-mobile.html").read_text(encoding="utf-8").replace(
            "/pet/", f"{self.prefix}/")
        for name in MOBILE_FILES:
            self.html = self.html.replace(f"/ui/{name}", f"{self.prefix}/ui/{name}")
        self.css = (ui_root / "pet-mobile.css").read_text(encoding="utf-8").replace(
            "/pet/", f"{self.prefix}/").encode()
        self.css_etag = '"' + hashlib.sha256(self.css).hexdigest() + '"'
        self.css_gzip = GzipAsset.prepare(self.css)
        self.gzip_assets = {}
        for name, path in sources.items():
            if path.suffix in {".js", ".css", ".json", ".moc3"} and path.is_file():
                compressed = GzipAsset.prepare(path.read_bytes())
                if compressed:
                    self.gzip_assets[name] = compressed
        self.core_url = f"{self.prefix}/{CORE_PATH}" if (pet_root / CORE_PATH).is_file() else CORE_CDN
        self.html = self.html.replace("window.VM_PET_ASSET_ROOT=",
                                      f"window.VM_PET_CORE_URL={json.dumps(self.core_url)};window.VM_PET_ASSET_ROOT=")
        self.html = self.html.replace("</head>", self._preloads(pet_root, vendor_root) + "</head>")

    def _preloads(self, pet_root: Path, vendor_root: Path) -> str:
        links = [(self.core_url, "script", False)]
        links.extend((f"{self.prefix}/vendor/{name}", "script", False)
                     for name, path in PET_VENDORS.items() if (vendor_root / path).is_file())
        runtime_links = len(links)
        model = pet_root / MODEL_PATH
        if model.is_file():
            # Invalid models remain renderer errors rather than server startup failures.
            try:
                settings = json.loads(model.read_text(encoding="utf-8"))
                refs = settings.get("FileReferences", {}) if isinstance(settings, dict) else {}
            except (ValueError, UnicodeError):
                refs = {}
            if not isinstance(refs, dict):
                refs = {}
            textures = refs.get("Textures", [])
            if not isinstance(textures, list):
                textures = []
            links.append((f"{self.prefix}/{MODEL_PATH}", "fetch", True))
            for name in (refs.get("Moc"), refs.get("Physics"), *textures):
                if not isinstance(name, str):
                    continue
                path = (model.parent / name).resolve()
                if path.is_file() and path.is_relative_to(model.parent.resolve()):
                    relative = path.relative_to(pet_root.resolve()).as_posix()
                    links.append((f"{self.prefix}/{relative}", "image" if name in textures else "fetch", True))
        tags = [f'<link rel="preload" href="{escape(url, quote=True)}" as="{kind}"'
                f'{" crossorigin" if cors else ""}>' for url, kind, cors in links]
        self.model_preloads = "".join(tags[runtime_links:])
        return "".join(tags)
