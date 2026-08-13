"""Print every setting the chamber controller reports over TCP."""

from tcp_common import request_settings

HOST = "192.168.88.248" #DEFAULT_HOST
PORT = 4321 #DEFAULT_PORT
TIMEOUT = 5.0 #DEFAULT_TIMEOUT


def main() -> None:
    settings = request_settings(host=HOST, port=PORT, timeout=TIMEOUT)

    print(f"Received {len(settings)} settings:")
    for key in sorted(settings.keys()):
        print(f"{key} = {settings[key]}")


if __name__ == "__main__":
    main()
