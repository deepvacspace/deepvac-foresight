from __future__ import annotations

from dataclasses import dataclass, asdict
from opcua import Client
import time
import json
import math

def limit(lo: float, x: float, hi: float) -> float:
    return max(lo, min(x, hi))


@dataclass
class PidValues:
    p: float = 0.0
    i: float = 0.0
    d: float = 0.0


@dataclass
class PidCoefs:
    p: float
    i: float
    d: float


class SimpleDiff:
    def __init__(self):
        self.prev_x = None
        self.prev_t = None
        self.out = 0.0  # "m_diff^.out"

    def update(self, x: float, t: float) -> float:
        if self.prev_x is None:
            self.prev_x = x
            self.prev_t = t
            self.out = 0.0
            return self.out

        dt = t - self.prev_t
        if dt <= 1e-9:
            self.out = 0.0
        else:
            self.out = (x - self.prev_x) / dt

        self.prev_x = x
        self.prev_t = t
        return self.out


class Pid:
    """
    - P part: (1/p_coef) * delta
    - I "reverse multiplier" when delta and I have opposite sign
    - I accumulation only when abs(delta) < 1.2 * p_coef and i_coef != 0
    - D part: (1/p_coef) * (d_coef * -diff_out)
    - clamp I to [u_min, u_max]
    - clamp D to [-0.4, 0.4]
    - clamp u to [u_min, u_max]
    """
    PID_I_REVERCE_MUL: float = 0.333

    def __init__(self, u_min: float, u_max: float):
        self.u_min = float(u_min)
        self.u_max = float(u_max)

        self.m_xManualPidSelect = False  
        self.m_pidCoefs = PidCoefs(p=1.0, i=1.0, d=0.0)  

        self.m_P_part = 0.0
        self.m_I_part = 0.0
        self.m_D_part = 0.0

        self.u = 0.0
        self.values = PidValues()

        self.diff = SimpleDiff()

    def set_coefs(self, p: float, i: float, d: float) -> None:
        self.m_pidCoefs = PidCoefs(p=float(p), i=float(i), d=float(d))

    def reset(self) -> None:
        self.m_P_part = 0.0
        self.m_I_part = 0.0
        self.m_D_part = 0.0
        self.u = 0.0
        self.values = PidValues()

        self.diff = SimpleDiff()

    def update(self, x_measured: float, x_target: float, enable: bool, now_t: float) -> tuple[float, PidValues]:
        if not enable:
            self.reset()
            return self.u, self.values

        delta = float(x_target) - float(x_measured)

        p_coef = self.m_pidCoefs.p
        i_coef = self.m_pidCoefs.i
        d_coef = self.m_pidCoefs.d

        if p_coef == 0.0:
            self.u = 0.0
            self.m_P_part = 0.0
            self.m_I_part = 0.0
            self.m_D_part = 0.0
            self.values = PidValues()
            return self.u, self.values

        self.m_P_part = (1.0 / p_coef) * delta

        if (delta * self.m_I_part) < 0.0:
            i_coef = i_coef * self.PID_I_REVERCE_MUL

        delta_edge = 1.2 * p_coef
        if (i_coef != 0.0) and (abs(delta) < delta_edge):
            self.m_I_part = self.m_I_part + (1.0 / p_coef) * (delta * 0.1 / i_coef)

        diff_out = self.diff.update(float(x_measured), now_t)
        self.m_D_part = (1.0 / p_coef) * (d_coef * (-diff_out))

        self.m_I_part = limit(self.u_min, self.m_I_part, self.u_max)
        self.m_D_part = limit(-0.4, self.m_D_part, 0.4)

        self.u = self.m_P_part + self.m_I_part + self.m_D_part
        self.u = limit(self.u_min, self.u, self.u_max)

        self.m_P_part = limit(self.u_min, self.m_P_part, self.u_max)

        self.values = PidValues(p=self.m_P_part, i=self.m_I_part, d=self.m_D_part)
        return self.u, self.values


def run_pid_from_opcua(
    endpoint: str,
    duration_s: float = 30.0,
    dt_s: float = 0.5,
    enable: bool = True,
    control_variable: str = "temp",
    temp_setpoint: float = 120.0,
    pres_setpoint: float = 1.0,
    p_coef: float = 10.0,
    i_coef: float = 30.0,
    d_coef: float = 0.0,
    u_min: float = 0.0,
    u_max: float = 100.0,
    output_json_path: str = "opcua_pid_log.json",
) -> None:
    client = Client(endpoint)
    client.connect()

    n_temp = client.get_node("ns=2;s=Testa chamber.temp")
    n_temp_raw = client.get_node("ns=2;s=Testa chamber.temp_raw")
    n_state = client.get_node("ns=2;s=Testa chamber.state")
    n_pres = client.get_node("ns=2;s=Testa chamber.pres")

    pid = Pid(u_min=u_min, u_max=u_max)
    pid.set_coefs(p=p_coef, i=i_coef, d=d_coef)

    logs: list[dict] = []
    t0 = time.time()

    try:
        while True:
            now = time.time()
            if (now - t0) >= duration_s:
                break

            temp = float(n_temp.get_value())
            temp_raw = float(n_temp_raw.get_value())
            state = n_state.get_value()
            pres = float(n_pres.get_value())

            if control_variable.lower() in ("temp", "temperature"):
                x_measured = temp
                x_target = temp_setpoint
            elif control_variable.lower() in ("pressure", "pres"):
                x_measured = pres
                x_target = pres_setpoint
            else:
                raise ValueError("control_variable must be 'temp' or 'pressure'.")

            u, parts = pid.update(
                x_measured=x_measured,
                x_target=x_target,
                enable=enable,
                now_t=now,
            )


            logs.append(
                {
                    "timestamp": now,
                    "state": state,
                    "temp": temp,
                    "temp_raw": temp_raw,
                    "pressure": pres,
                    "control_variable": control_variable,
                    "x_target": x_target,
                    "x_measured": x_measured,
                    "delta": x_target - x_measured,
                    "u": u,
                    "pid_parts": asdict(parts),
                    "pid_coefs": {"p": p_coef, "i": i_coef, "d": d_coef},
                    "u_limits": {"min": u_min, "max": u_max},
                }
            )

            time.sleep(dt_s)

    finally:
        client.disconnect()

    with open(output_json_path, "w") as f:
        json.dump(logs, f, indent=2)

    print(f"Saved PID log to: {output_json_path}")


def main() -> None:
    run_pid_from_opcua(
        endpoint="opc.tcp://192.168.88.166:12345",
        duration_s=30.0,
        dt_s=0.5,
        enable=True,
        control_variable="temp",
        temp_setpoint=120.0,
        pres_setpoint=1.0,
        p_coef=10.0,
        i_coef=30.0,
        d_coef=0.0,
        u_min=0.0,
        u_max=100.0,
        output_json_path="opcua_pid_log.json",
    )


if __name__ == "__main__":
    main()
