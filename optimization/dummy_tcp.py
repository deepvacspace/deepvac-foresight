import socket
import time
from typing import Optional, Dict
from opcua import Client
import json

from tcp.tcp_common import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    INCOMING_STATE_VALUES_CODE,
    DEFAULT_TIMEOUT,
    make_packet,
    parse_state_values,
    read_one_packet,
    make_settings_map_body,
    request_settings,
    send_no_wait,
    setpoint_job_body,
)

HOST = DEFAULT_HOST
PORT = DEFAULT_PORT
TIMEOUT = 10.0

def extract_temp_pid_arrays(settings: Dict[str, float]):
    p_list = [settings[f"p1p{i}"] for i in range(5)]
    i_list = [settings[f"p1i{i}"] for i in range(5)]
    d_list = [settings[f"p1d{i}"] for i in range(5)]
    return p_list, i_list, d_list


def build_temp_pid_array_settings(p_list, i_list, d_list) -> Dict[str, float]:
    if len(p_list) != 5 or len(i_list) != 5 or len(d_list) != 5:
        raise ValueError("p_list, i_list, d_list must each have exactly 5 values")

    settings: Dict[str, float] = {}
    for row in range(5):
        settings[f"p1p{row}"] = float(p_list[row])
        settings[f"p1i{row}"] = float(i_list[row])
        settings[f"p1d{row}"] = float(d_list[row])
    return settings


def write_full_p1_no_reply(
    settings: Dict[str, float],
    host: str = HOST,
    port: int = PORT,
    timeout: float = TIMEOUT,
) -> None:
    body = make_settings_map_body(settings)
    send_no_wait(body, host=host, port=port, timeout=timeout)


def print_temp_pid_array(title: str, settings: Dict[str, float]) -> None:
    print(title)
    for i in range(5):
        print(
            f"row {i}: "
            f"kp={settings.get(f'p1p{i}')}, "
            f"ki={settings.get(f'p1i{i}')}, "
            f"kd={settings.get(f'p1d{i}')}"
        )
    print()


def replace_one_temp_pid_row_and_verify(
    row: int,
    kp: float,
    ki: float,
    kd: float,
    host: str = HOST,
    port: int = PORT,
    timeout: float = TIMEOUT,
    settle_s: float = 0.5,
) -> None:
    if not (0 <= row <= 4):
        raise ValueError("row must be between 0 and 4")

    before = request_settings(host=host, port=port, timeout=timeout)
    print_temp_pid_array("Current p1 PID array:", before)

    p_list, i_list, d_list = extract_temp_pid_arrays(before)
    p_list[row] = float(kp)
    i_list[row] = float(ki)
    d_list[row] = float(kd)

    to_send = build_temp_pid_array_settings(p_list, i_list, d_list)

    print(f"Writing full p1 array; modified row {row}:")
    print(f"  p1p{row} = {kp}")
    print(f"  p1i{row} = {ki}")
    print(f"  p1d{row} = {kd}")
    print()

    # Important: do not wait for reply on write
    write_full_p1_no_reply(to_send, host=host, port=port, timeout=timeout)

    time.sleep(settle_s)

    after = request_settings(host=host, port=port, timeout=timeout)
    print_temp_pid_array("Readback p1 PID array after write:", after)

    ok = (
        abs(after[f"p1p{row}"] - float(kp)) < 1e-6
        and abs(after[f"p1i{row}"] - float(ki)) < 1e-6
        and abs(after[f"p1d{row}"] - float(kd)) < 1e-6
    )

    if ok:
        print(f"SUCCESS: row {row} matches requested values.")
    else:
        print(f"WARNING: row {row} does not match requested values.")
        print(
            f"Expected: kp={kp}, ki={ki}, kd={kd}\n"
            f"Got:      kp={after[f'p1p{row}']}, "
            f"ki={after[f'p1i{row}']}, "
            f"kd={after[f'p1d{row}']}"
        )



TEMP_REF_INDEX = 2

# For a simple manual temp setpoint, use control_params defaults.
DEFAULT_FLAGS = 3


def read_temp_ref_once(sock: socket.socket, timeout_s: float = 15.0) -> Optional[float]:
    sock.settimeout(timeout_s)
    while True:
        body = read_one_packet(sock)
        if len(body) < 1 or body[0] != INCOMING_STATE_VALUES_CODE:
            continue
        values = parse_state_values(body)
        if TEMP_REF_INDEX < len(values):
            return values[TEMP_REF_INDEX]


def read_latest_temp_ref(sock: socket.socket, reads: int = 2, timeout_s: float = 5.0) -> Optional[float]:
    """
    Read a small number of subsequent STATE_VALUES packets and keep the latest temp_ref.
    This avoids trusting the very next packet after the write.
    """
    latest = None
    sock.settimeout(timeout_s)

    for _ in range(reads):
        body = read_one_packet(sock)
        if len(body) < 1 or body[0] != INCOMING_STATE_VALUES_CODE:
            continue
        values = parse_state_values(body)
        if TEMP_REF_INDEX < len(values):
            latest = values[TEMP_REF_INDEX]

    return latest


def main() -> None:
    new_temp = 24.0
    duration_s = 30
    flags = DEFAULT_FLAGS

    client = Client("opc.tcp://192.168.88.174:12345") # Real PC 192.168.88.144:12345
    client.connect()

    readings = []

    reads = {
        "timestamp": time.time(),
        "temp": client.get_node("ns=2;s=Testa chamber.temp").get_value(),
        "temp_raw": client.get_node("ns=2;s=Testa chamber.temp_raw").get_value(),
        "temp_ref": client.get_node("ns=2;s=Testa chamber.temp_ref").get_value(),
        "state": client.get_node("ns=2;s=Testa chamber.state").get_value(),
        "temp_u": client.get_node("ns=2;s=Testa chamber.temp_u").get_value(),
        "temp_u_p": client.get_node("ns=2;s=Testa chamber.temp_u_p").get_value(),
        "temp_u_i": client.get_node("ns=2;s=Testa chamber.temp_u_i").get_value(),
        "temp_u_d": client.get_node("ns=2;s=Testa chamber.temp_u_d").get_value(),
        "temp_kp": client.get_node("ns=2;s=Testa chamber.temp_kp").get_value(),
        "temp_ki": client.get_node("ns=2;s=Testa chamber.temp_ki").get_value(),
        "temp_kd": client.get_node("ns=2;s=Testa chamber.temp_kd").get_value(),
        "temp_pid_idx": client.get_node("ns=2;s=Testa chamber.temp_pid_idx").get_value()
    }

    readings.append(reads)
    time.sleep(0.5)

    json_data = json.dumps(readings, indent=2)
    print(json_data)



    with socket.create_connection((HOST, PORT), timeout=TIMEOUT) as sock:
        sock.settimeout(TIMEOUT)

        before = read_temp_ref_once(sock, timeout_s=15.0)
        print("temp_ref before =", before)

        body = setpoint_job_body(
            temp_c=new_temp,
            duration_s=duration_s,
            flags=flags,
        )
        sock.sendall(make_packet(body))
        print(f"Sent one-interval SETPOINT job: temp={new_temp}, duration={duration_s}, flags={flags}")

        after = read_latest_temp_ref(sock, reads=2, timeout_s=10.0)
        print("temp_ref after  =", after)

        if after is not None and abs(after - new_temp) < 1e-4:
            print("SUCCESS: temp_ref changed to requested value.")
        else:
            print("WARNING: temp_ref did not match the requested value.")
    
    replace_one_temp_pid_row_and_verify(
        row=2,
        kp=6.0,
        ki=950.0,
        kd=12.0,
    )

    readings = []

    start_time = time.time()
    
    while time.time() - start_time < 5.0:
        reads = {
            "timestamp": time.time(),
            "temp": client.get_node("ns=2;s=Testa chamber.temp").get_value(),
            "temp_raw": client.get_node("ns=2;s=Testa chamber.temp_raw").get_value(),
            "temp_ref": client.get_node("ns=2;s=Testa chamber.temp_ref").get_value(),
            "state": client.get_node("ns=2;s=Testa chamber.state").get_value(),
            "temp_u": client.get_node("ns=2;s=Testa chamber.temp_u").get_value(),
            "temp_u_p": client.get_node("ns=2;s=Testa chamber.temp_u_p").get_value(),
            "temp_u_i": client.get_node("ns=2;s=Testa chamber.temp_u_i").get_value(),
            "temp_u_d": client.get_node("ns=2;s=Testa chamber.temp_u_d").get_value(),
            "temp_kp": client.get_node("ns=2;s=Testa chamber.temp_kp").get_value(),
            "temp_ki": client.get_node("ns=2;s=Testa chamber.temp_ki").get_value(),
            "temp_kd": client.get_node("ns=2;s=Testa chamber.temp_kd").get_value(),
            "temp_pid_idx": client.get_node("ns=2;s=Testa chamber.temp_pid_idx").get_value()
        }
        readings.append(reads)
        time.sleep(0.5)

        json_data = json.dumps(readings, indent=2)
        print(json_data)

        with open("test_read.json", "w") as f:
            json.dump(readings, f, indent=2)

        client.disconnect()


if __name__ == "__main__":
    main()
