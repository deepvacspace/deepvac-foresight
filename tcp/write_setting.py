import time
from typing import Dict

from tcp.tcp_common import (
        DEFAULT_HOST,
        DEFAULT_PORT,
        DEFAULT_TIMEOUT,
        make_settings_map_body,
        request_settings,
        send_no_wait,
)

HOST = DEFAULT_HOST
PORT = DEFAULT_PORT
TIMEOUT = DEFAULT_TIMEOUT

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


def main() -> None:
    replace_one_temp_pid_row_and_verify(
        row=1,
        kp=5.0,
        ki=900.0,
        kd=10.0,
    )


if __name__ == "__main__":
    main()
