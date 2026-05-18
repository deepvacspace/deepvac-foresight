#!/usr/bin/env python3
"""Shared OPC communication helpers."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import pandas as pd
from opcua import Client


@dataclass
class OPCNodeMap:
    temp: str
    temp_ref: str
    temp_raw: str
    temp_u: str
    temp_u_p: str
    temp_u_i: str
    temp_u_d: str


def _timestamp_now() -> float:
    return time.time()


def _make_run_id(prefix: str = "run") -> str:
    return f"{prefix}_{int(time.time())}_{uuid.uuid4().hex[:8]}"


def read_opc_snapshot(client: Client, nodes: OPCNodeMap) -> Dict[str, float]:
    return {
        "temp": float(client.get_node(nodes.temp).get_value()),
        "temp_ref": float(client.get_node(nodes.temp_ref).get_value()),
        "temp_raw": float(client.get_node(nodes.temp_raw).get_value()),
        "temp_u": float(client.get_node(nodes.temp_u).get_value()),
        "temp_u_p": float(client.get_node(nodes.temp_u_p).get_value()),
        "temp_u_i": float(client.get_node(nodes.temp_u_i).get_value()),
        "temp_u_d": float(client.get_node(nodes.temp_u_d).get_value()),
    }


def run_opc_test(
    endpoint: str,
    nodes: OPCNodeMap,
    kp: float,
    ki: float,
    kd: float,
    duration_s: float,
    dt_s: float,
    run_id: Optional[str] = None,
    progress_every_s: float = 30.0,
    verbose: bool = True,
    read_retries: int = 2,
    read_retry_delay_s: float = 0.25,
    max_consecutive_failures: int = 10,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    if dt_s <= 0:
        raise ValueError("dt_s must be > 0")
    if duration_s <= 0:
        raise ValueError("duration_s must be > 0")
    if progress_every_s < 0:
        raise ValueError("progress_every_s must be >= 0")
    if read_retries < 0:
        raise ValueError("read_retries must be >= 0")
    if read_retry_delay_s < 0:
        raise ValueError("read_retry_delay_s must be >= 0")
    if max_consecutive_failures <= 0:
        raise ValueError("max_consecutive_failures must be > 0")

    run_id = run_id or _make_run_id()
    client = Client(endpoint)
    connected = False
    rows = []
    t0 = _timestamp_now()
    next_progress_ts = t0 + progress_every_s if progress_every_s > 0 else float("inf")
    sq_error_sum = 0.0
    consecutive_failures = 0

    def read_with_retries() -> Dict[str, float]:
        last_exc: Optional[Exception] = None
        for attempt in range(read_retries + 1):
            try:
                return read_opc_snapshot(client, nodes)
            except Exception as exc:
                last_exc = exc
                if attempt < read_retries and read_retry_delay_s > 0:
                    time.sleep(read_retry_delay_s)
        assert last_exc is not None
        raise last_exc

    try:
        try:
            client.connect()
            connected = True
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to OPC endpoint '{endpoint}'. "
                "Verify server is running, reachable, and listening on this port."
            ) from exc

        if verbose:
            print(
                f"[run {run_id}] connected, sampling every {dt_s:.3f}s for {duration_s:.1f}s"
            )

        while True:
            now = _timestamp_now()
            if now - t0 >= duration_s:
                break

            try:
                snap = read_with_retries()
                consecutive_failures = 0
            except Exception as exc:
                consecutive_failures += 1
                if verbose:
                    elapsed = now - t0
                    print(
                        f"[run {run_id}] read timeout/error at {elapsed:.1f}s "
                        f"(consecutive_failures={consecutive_failures}): {exc}"
                    )

                if consecutive_failures >= max_consecutive_failures:
                    raise RuntimeError(
                        f"Too many consecutive OPC read failures ({consecutive_failures}). "
                        "Stopping run to avoid logging invalid data."
                    ) from exc

                time.sleep(dt_s)
                continue

            sq_error = float((snap["temp_ref"] - snap["temp"]) ** 2)
            sq_error_sum += sq_error
            rows.append(
                {
                    "run_id": run_id,
                    "timestamp": now,
                    "kp": float(kp),
                    "ki": float(ki),
                    "kd": float(kd),
                    **snap,
                    "sq_error": sq_error,
                }
            )

            if verbose and now >= next_progress_ts:
                elapsed = now - t0
                remaining = max(duration_s - elapsed, 0.0)
                n = len(rows)
                running_mse = sq_error_sum / n
                print(
                    f"[run {run_id}] samples={n} elapsed={elapsed:.1f}s "
                    f"remaining={remaining:.1f}s temp={snap['temp']:.3f} "
                    f"temp_ref={snap['temp_ref']:.3f} mse={running_mse:.6f}"
                )
                next_progress_ts += progress_every_s

            time.sleep(dt_s)
    finally:
        if connected:
            try:
                client.disconnect()
            except Exception:
                # Ignore shutdown errors to preserve the original failure context.
                pass

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No OPC samples were collected. Check connectivity and test timing.")

    mse = float(df["sq_error"].mean())
    summary = {
        "run_id": run_id,
        "start_ts": float(df["timestamp"].iloc[0]),
        "end_ts": float(df["timestamp"].iloc[-1]),
        "duration_s": float(df["timestamp"].iloc[-1] - df["timestamp"].iloc[0]),
        "num_samples": int(len(df)),
        "kp": float(kp),
        "ki": float(ki),
        "kd": float(kd),
        "mse": mse,
    }
    return df, summary
