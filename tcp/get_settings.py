from tcp_common import (
        DEFAULT_HOST,
        DEFAULT_PORT,
        DEFAULT_TIMEOUT,
        request_settings
)

HOST = DEFAULT_HOST
PORT = DEFAULT_PORT
TIMEOUT = DEFAULT_TIMEOUT


def main() -> None:
    settings = request_settings(host=HOST, port=PORT, timeout=TIMEOUT)

    print(f"Received {len(settings)} settings:")
    for key in sorted(settings.keys()):
        print(f"{key} = {settings[key]}")


if __name__ == "__main__":
    main()