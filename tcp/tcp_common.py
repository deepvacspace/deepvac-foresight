from contextlib import contextmanager
import socket
import struct
import time
from typing import Any, Dict, Iterator, List, Optional, Tuple

DEFAULT_HOST = "192.168.88.188"
DEFAULT_PORT = 4321
DEFAULT_TIMEOUT = 5.0
DEFAULT_JOB_FLAGS = 3

HEADER = b"\xAA\x55\xBB\x77"

# Text command
TEXT_CMD_CODE = 0x01

# App -> controller
OUTGOING_SETTINGS_MAP_CODE = 0x02
OUTGOING_JOB_CODE = 0x03
PROGTYPE_SETPOINT = 1

# Controller -> app
INCOMING_STATE_VALUES_CODE = 0x01
INCOMING_STATE_NAMES_CODE = 0x02
INCOMING_SETTINGS_MAP_CODE = 0x03
INCOMING_CONFIGURATION_CODE = 0x04
INCOMING_ALARM_CODE = 0x05
INCOMING_JOB_FINISHED_CODE = 0x06
INCOMING_USER_MSG_CODE = 0x07

_STATE_NAMES_CACHE: List[str] | None = None


# -------------------------- TCP PROTOCOL UTILITIES --------------------------

def crc16_ccitt_false(data: bytes) -> int:
    poly = 0x1021
    crc = 0xFFFF

    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) & 0xFFFF) ^ poly
            else:
                crc = (crc << 1) & 0xFFFF

    return crc & 0xFFFF


def make_pascal_string(text: str) -> bytes:
    raw = text.encode("utf-8")
    if len(raw) > 255:
        raise ValueError(f"String too long: {text}")
    return struct.pack("<B", len(raw)) + raw


def make_text_cmd_body(cmd: str) -> bytes:
    return bytes([TEXT_CMD_CODE]) + make_pascal_string(cmd)


def make_settings_map_body(settings: Dict[str, float], msg_code: int = OUTGOING_SETTINGS_MAP_CODE) -> bytes:
    if len(settings) > 255:
        raise ValueError("Too many settings")

    body = bytearray([msg_code, len(settings)])

    for key, value in settings.items():
        body += make_pascal_string(key)
        body += struct.pack("<f", float(value))

    return bytes(body)


def setpoint_job_body(temp_c: float, duration_s: int, flags: int) -> bytes:
    data = bytearray()
    data.append(OUTGOING_JOB_CODE)
    data.append(PROGTYPE_SETPOINT)
    data += struct.pack("<H", 0)  # intervalIndex
    data += struct.pack("<H", 0)  # cycle_count
    data += struct.pack("<H", 0)  # cycle_start
    data += struct.pack("<H", 0)  # cycle_end
    data += struct.pack("<H", 1)  # intervalsCount
    data += struct.pack("<I", int(duration_s))
    data += struct.pack("<f", float(temp_c))
    data += struct.pack("<I", int(flags))
    return bytes(data)

def make_packet(body: bytes) -> bytes:
    packet_wo_header = bytearray()
    packet_wo_header += struct.pack("<H", len(body))
    packet_wo_header += body

    crc = crc16_ccitt_false(bytes(packet_wo_header))

    # Match StreamBuffer::makePackage(): CRC high byte first, then low byte
    packet_wo_header.append((crc >> 8) & 0xFF)
    packet_wo_header.append(crc & 0xFF)

    return HEADER + bytes(packet_wo_header)

@contextmanager
def connected_socket(
    body: bytes,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
) -> Iterator[socket.socket]:
    packet = make_packet(body)

    with socket.create_connection((host, port), timeout=timeout) as sock:
        sock.settimeout(timeout)
        sock.sendall(packet)
        yield sock


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        chunk = sock.recv(n - len(chunks))
        if not chunk:
            raise ConnectionError("Socket closed while reading data")
        chunks += chunk
    return bytes(chunks)

def send_and_wait(
    body: bytes,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
) -> bytes:
    with connected_socket(body, host=host, port=port, timeout=timeout) as sock:
        return read_one_packet(sock)


def send_no_wait(
    body: bytes,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    with connected_socket(body, host=host, port=port, timeout=timeout):
        pass


def read_one_packet(sock: socket.socket) -> bytes:
    header = recv_exact(sock, 4)
    if header != HEADER:
        raise ValueError(f"Bad header: {header.hex(' ')}")

    size_bytes = recv_exact(sock, 2)
    body_size = struct.unpack("<H", size_bytes)[0]

    body = recv_exact(sock, body_size)
    crc_bytes = recv_exact(sock, 2)

    crc_check = crc16_ccitt_false(size_bytes + body + crc_bytes)
    if crc_check != 0:
        raise ValueError(f"CRC check failed: 0x{crc_check:04X}")

    return body


def parse_pascal_string(data: bytes, offset: int) -> Tuple[str, int]:
    if offset >= len(data):
        raise ValueError("Offset out of range while parsing Pascal string")

    strlen = data[offset]
    offset += 1

    if offset + strlen > len(data):
        raise ValueError("Pascal string length exceeds available data")

    text = data[offset : offset + strlen].decode("utf-8", errors="replace")
    offset += strlen
    return text, offset

def parse_settings_map_body(
    body: bytes,
    expected_msg_code: int = INCOMING_SETTINGS_MAP_CODE,
    warn_trailing: bool = False,
) -> Dict[str, float]:
    if len(body) < 2:
        raise ValueError("Body too short")

    msg_code = body[0]
    if msg_code != expected_msg_code:
        raise ValueError(f"Unexpected message code: 0x{msg_code:02X}")

    count = body[1]
    offset = 2
    settings: Dict[str, float] = {}

    for _ in range(count):
        key, offset = parse_pascal_string(body, offset)

        if offset + 4 > len(body):
            raise ValueError("Not enough bytes for float value")

        value = struct.unpack("<f", body[offset : offset + 4])[0]
        offset += 4
        settings[key] = value

    if warn_trailing and offset != len(body):
        print(f"Warning: {len(body) - offset} trailing bytes left after parsing")

    return settings


def parse_state_names(body: bytes, expected_msg_code: int = INCOMING_STATE_NAMES_CODE) -> List[str]:
    if len(body) < 2 or body[0] != expected_msg_code:
        raise ValueError("Not a STATE_NAMES packet")

    count = body[1]
    offset = 2
    names: List[str] = []

    for _ in range(count):
        name, offset = parse_pascal_string(body, offset)
        names.append(name)

    return names


def parse_state_values(body: bytes, expected_msg_code: int = INCOMING_STATE_VALUES_CODE) -> List[float]:
    if len(body) < 2 or body[0] != expected_msg_code:
        raise ValueError("Not a STATE_VALUES packet")

    count = body[1]
    raw = body[2:]

    expected = count * 4
    if len(raw) < expected:
        raise ValueError("STATE_VALUES payload too short")

    return list(struct.unpack("<" + "f" * count, raw[:expected]))

# -------------------------- UTILITIES ---------------------------

def _pid_keys(row: int) -> Tuple[str, str, str]:
    return f"p1p{row}", f"p1i{row}", f"p1d{row}"

# -------------------------- TCP FUNCTIONS --------------------------

def publish_temp_ref_job(
    temp_ref: float,
    duration_s: float,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
    flags: int = DEFAULT_JOB_FLAGS,
) -> None:
    body = setpoint_job_body(temp_c=temp_ref, duration_s=int(duration_s), flags=flags)
    send_no_wait(body, host=host, port=port, timeout=timeout)

def write_pid_row(
    row: int,
    kp: int,
    ki: int,
    kd: int,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
) -> None:
    kp_key, ki_key, kd_key = _pid_keys(row)
    body = make_settings_map_body(
        {
            kp_key: float(kp),
            ki_key: float(ki),
            kd_key: float(kd),
        }
    )
    send_no_wait(body, host=host, port=port, timeout=timeout)


def verify_pid(
    row: int,
    args: Any,
    expected_kp: int,
    expected_ki: int,
    expected_kd: int,
    tol: float = 1e-3,
    max_attempts: int = 5,
    delay_s: float = 0.5,
) -> None:
    kp_key, ki_key, kd_key = _pid_keys(row)
    last_actual: Optional[Dict[str, float]] = None

    for attempt in range(max_attempts):
        settings = request_settings(
            host=args.tcp_host,
            port=args.tcp_port,
            timeout=args.tcp_timeout,
        )
        for key in (kp_key, ki_key, kd_key):
            if key not in settings:
                raise KeyError(f"TCP settings missing expected key: {key}")

        last_actual = {
            "kp": float(settings[kp_key]),
            "ki": float(settings[ki_key]),
            "kd": float(settings[kd_key]),
        }
        if (
            abs(last_actual["kp"] - float(expected_kp)) <= tol
            and abs(last_actual["ki"] - float(expected_ki)) <= tol
            and abs(last_actual["kd"] - float(expected_kd)) <= tol
        ):
            return
        if attempt < max_attempts - 1:
            time.sleep(max(0.0, delay_s))

    if last_actual is None:
        raise RuntimeError("PID verify could not read values from TCP settings")
    raise RuntimeError(
        "PID verify failed: "
        f"expected kp={expected_kp}, ki={expected_ki}, kd={expected_kd}; "
        f"got kp={last_actual['kp']}, ki={last_actual['ki']}, kd={last_actual['kd']}"
    )


def apply_pid_update(
    label: str,
    run_id: str,
    row: int,
    pid: Tuple[int, int, int],
    args: Any,
    events: list,
) -> Tuple[int, int, int]:
    kp, ki, kd = pid
    write_pid_row(
        row=row,
        kp=kp,
        ki=ki,
        kd=kd,
        host=args.tcp_host,
        port=args.tcp_port,
        timeout=args.tcp_timeout,
    )
    verify_pid(
        row=row,
        args=args,
        expected_kp=kp,
        expected_ki=ki,
        expected_kd=kd,
    )
    print(f"[run {run_id}] PID update ({label}): kp={kp}, ki={ki}, kd={kd}")
    events.append({"event": label, "kp": kp, "ki": ki, "kd": kd, "timestamp": time.time()})
    return kp, ki, kd

def request_settings(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
    expected_msg_code: int = INCOMING_SETTINGS_MAP_CODE,
    max_wait_s: float = 20.0,
    max_packets: int = 500,
) -> Dict[str, float]:
    start = time.time()
    skipped_codes: List[int] = []

    with connected_socket(
        make_text_cmd_body("get_settings"),
        host=host,
        port=port,
        timeout=timeout,
    ) as sock:
        packets = 0
        while packets < max_packets and (time.time() - start) <= max_wait_s:
            try:
                reply_body = read_one_packet(sock)
            except socket.timeout:
                continue

            packets += 1
            if len(reply_body) >= 1 and reply_body[0] == expected_msg_code:
                return parse_settings_map_body(reply_body, expected_msg_code=expected_msg_code)
            if len(reply_body) >= 1:
                skipped_codes.append(reply_body[0])

    skipped = ", ".join(f"0x{code:02X}" for code in skipped_codes[:10])
    if len(skipped_codes) > 10:
        skipped += ", ..."
    raise TimeoutError(
        "Timed out waiting for SETTINGS_MAP packet after get_settings request. "
        f"Skipped message codes: [{skipped}]"
    )


def request_states(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
    max_wait_s: float = 20.0,
    max_packets: int = 500,
) -> Dict[str, float]:
    global _STATE_NAMES_CACHE

    start = time.time()

    with connected_socket(
        make_text_cmd_body("get_states"),
        host=host,
        port=port,
        timeout=timeout,
    ) as sock:
        names = _STATE_NAMES_CACHE[:] if _STATE_NAMES_CACHE is not None else None
        values = None
        packets = 0

        while (
            (names is None or values is None)
            and packets < max_packets
            and (time.time() - start) <= max_wait_s
        ):
            try:
                body = read_one_packet(sock)
            except socket.timeout:
                if names is not None and values is not None:
                    break
                continue

            packets += 1
            msg_code = body[0]

            if msg_code == INCOMING_STATE_NAMES_CODE:
                names = parse_state_names(body)
                _STATE_NAMES_CACHE = names[:]
            elif msg_code == INCOMING_STATE_VALUES_CODE:
                values = parse_state_values(body)

        if names is None and _STATE_NAMES_CACHE is not None:
            names = _STATE_NAMES_CACHE[:]

        if names is None or values is None:
            raise TimeoutError(
                "Timed out waiting for get_states reply "
                f"(have_names={names is not None}, have_values={values is not None}, "
                f"packets={packets})"
            )

        if len(names) != len(values):
            raise ValueError(
                f"State names count ({len(names)}) does not match state values count ({len(values)})"
            )

        return dict(zip(names, values))


def request_temperature_states(
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict[str, float]:
    states = request_states(host=host, port=port, timeout=timeout)

    aliases = {
        "temp": ("temp",),
        "temp_ref": ("temp_ref",),
        "temp_raw": ("temp_raw",),
        "temp_u": ("temp_u",),
        "temp_u_p": ("temp_u_p",),
        "temp_u_i": ("temp_u_i",),
        "temp_u_d": ("temp_u_d",),
        "kp": ("temp_kp", "kp"),
        "ki": ("temp_ki", "ki"),
        "kd": ("temp_kd", "kd"),
    }

    relevant: Dict[str, float] = {}
    missing: List[str] = []

    for out_key, candidates in aliases.items():
        found = None
        for candidate in candidates:
            if candidate in states:
                found = float(states[candidate])
                break
        if found is None:
            missing.append(out_key)
        else:
            relevant[out_key] = found

    if missing:
        raise KeyError(f"Missing relevant TCP states: {missing}")

    return relevant
