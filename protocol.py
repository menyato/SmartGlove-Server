"""
protocol.py — framed JSON over TCP (server side).

4-byte big-endian length prefix + UTF-8 JSON body. Matches the OrangePi
ServerLink and the original client/server wire format exactly.
"""

import json
import struct


def recv_exact(sock, n: int) -> bytes | None:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf


def recv_msg(sock) -> dict | None:
    raw_len = recv_exact(sock, 4)
    if raw_len is None:
        return None
    n = struct.unpack(">I", raw_len)[0]
    raw = recv_exact(sock, n)
    if raw is None:
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        print(f"[PROTO] recv_msg: malformed payload ({e}) — dropping message")
        return None


def send_msg(sock, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    try:
        sock.sendall(struct.pack(">I", len(data)) + data)
    except OSError as e:
        raise ConnectionError(f"send_msg failed: {e}") from e