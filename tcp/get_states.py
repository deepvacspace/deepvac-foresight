from tcp_common import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    DEFAULT_TIMEOUT,
    request_states,
)

HOST = DEFAULT_HOST
PORT = DEFAULT_PORT
TIMEOUT = DEFAULT_TIMEOUT


def main() -> None:
    states = request_states(host=HOST, port=PORT, timeout=TIMEOUT)

    print(f"Received {len(states)} states:")
    for key in sorted(states.keys()):
        print(f"{key} = {states[key]}")


if __name__ == "__main__":
    main()