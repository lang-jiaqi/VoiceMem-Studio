"""Browser-origin checks shared by HTTP mutations and WebSocket handshakes."""
from urllib.parse import urlsplit


def same_origin(connection) -> bool:
    """Allow same-origin browsers and clients that omit the Origin header.

    Native clients remain supported. Opaque, malformed and cross-origin browser
    origins are rejected; deployment authentication remains separately owned.
    """
    origin = connection.headers.get("origin")
    if origin is None:
        return True
    if origin != origin.strip() or any(ord(char) <= 32 or ord(char) == 127 for char in origin):
        return False
    try:
        source = urlsplit(origin)
        target = urlsplit(str(connection.base_url))
        scheme = {"ws": "http", "wss": "https"}.get(target.scheme, target.scheme)
        if (source.scheme not in {"http", "https"} or not source.hostname
                or source.username is not None or source.password is not None
                or source.path or source.query or source.fragment):
            return False
        source_port = source.port or (443 if source.scheme == "https" else 80)
        target_port = target.port or (443 if scheme == "https" else 80)
        return (source.scheme, source.hostname, source_port) == (scheme, target.hostname, target_port)
    except ValueError:
        return False
