"""Unit tests for the chamber TCP wire protocol (tcp/tcp_common.py, imported
via deepvac.protocol -- the canonical path per DEV_GUIDE.md §9). Pure framing
logic: no socket, no hardware."""

from __future__ import annotations

import struct

import pytest
from deepvac import protocol


def test_crc16_ccitt_false_known_vector():
    # "123456789" is the standard CRC-16/CCITT-FALSE test vector; the
    # expected residue 0x29B1 is documented for this exact variant.
    assert protocol.crc16_ccitt_false(b"123456789") == 0x29B1


def test_crc16_is_sensitive_to_every_byte():
    base = protocol.crc16_ccitt_false(b"\x00\x01\x02\x03")
    for i in range(4):
        mutated = bytearray(b"\x00\x01\x02\x03")
        mutated[i] ^= 0xFF
        assert protocol.crc16_ccitt_false(bytes(mutated)) != base


def test_make_pascal_string_round_trips():
    encoded = protocol.make_pascal_string("p1p0")
    text, offset = protocol.parse_pascal_string(encoded, 0)
    assert text == "p1p0"
    assert offset == len(encoded)


def test_make_pascal_string_rejects_too_long():
    with pytest.raises(ValueError):
        protocol.make_pascal_string("x" * 256)


def test_parse_pascal_string_rejects_truncated_length():
    # length byte claims 10 chars but only 2 follow
    with pytest.raises(ValueError):
        protocol.parse_pascal_string(bytes([10, ord("a"), ord("b")]), 0)


def test_settings_map_round_trips():
    settings = {"p1p0": 6.0, "p1i0": 997.0, "p1d0": 16.0}
    body = protocol.make_settings_map_body(settings, msg_code=protocol.INCOMING_SETTINGS_MAP_CODE)
    parsed = protocol.parse_settings_map_body(body, expected_msg_code=body[0])
    assert parsed.keys() == settings.keys()
    for key, value in settings.items():
        assert parsed[key] == pytest.approx(value, rel=1e-6)


def test_settings_map_rejects_wrong_msg_code():
    body = protocol.make_settings_map_body({"p1p0": 1.0})
    with pytest.raises(ValueError):
        protocol.parse_settings_map_body(body, expected_msg_code=body[0] + 1)


def test_make_packet_header_and_length_prefix():
    body = b"\x01\x02\x03"
    packet = protocol.make_packet(body)
    assert packet[:4] == b"\xAA\x55\xBB\x77"
    (length,) = struct.unpack("<H", packet[4:6])
    assert length == len(body)
    assert packet[6 : 6 + len(body)] == body


def test_make_packet_crc_matches_crc16_over_length_and_body():
    body = b"hello"
    packet = protocol.make_packet(body)
    length_and_body = packet[4:6] + body
    expected_crc = protocol.crc16_ccitt_false(length_and_body)
    crc_hi, crc_lo = packet[-2], packet[-1]
    assert (crc_hi << 8) | crc_lo == expected_crc


def test_pid_keys_are_row_specific():
    assert protocol._pid_keys(0) == ("p1p0", "p1i0", "p1d0")
    assert protocol._pid_keys(2) == ("p1p2", "p1i2", "p1d2")


def test_parse_state_values_round_trips_via_struct():
    values = [1.5, -2.25, 100.0]
    body = bytes([protocol.INCOMING_STATE_VALUES_CODE, len(values)])
    for v in values:
        body += struct.pack("<f", v)
    parsed = protocol.parse_state_values(body, expected_msg_code=body[0])
    assert parsed == pytest.approx(values, rel=1e-6)
