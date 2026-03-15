import os
import sys
import socket
import threading
import time
import webbrowser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template
from scapy.all import sniff, IP, TCP, UDP, get_if_list, get_if_addr


def resource_path(relative_path: str) -> str:
    """
    Resolve resource paths for normal Python runs and PyInstaller builds.
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)


MODEL_PATH = Path(resource_path("saved_model/anomaly_model.pkl"))
TEMPLATE_DIR = resource_path("templates")

app = Flask(__name__, template_folder=TEMPLATE_DIR)
pipeline = joblib.load(MODEL_PATH)

SENSITIVE_PORTS = {21, 22, 23, 53, 80, 135, 137, 138, 139, 443, 445, 1433, 3306, 3389}


def detect_active_interface() -> str | None:
    """
    Auto-detect the active network interface by matching the outbound local IP
    to a Scapy interface.
    """
    local_ip = None

    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.connect(("8.8.8.8", 80))
        local_ip = sock.getsockname()[0]
        sock.close()
    except Exception:
        local_ip = None

    if local_ip:
        for iface in get_if_list():
            try:
                iface_ip = get_if_addr(iface)
                if iface_ip == local_ip:
                    return iface
            except Exception:
                continue

    for iface in get_if_list():
        try:
            iface_ip = get_if_addr(iface)
            if iface_ip and iface_ip != "0.0.0.0" and not iface_ip.startswith("127."):
                return iface
        except Exception:
            continue

    return None


def get_flow_key(packet):
    if IP not in packet:
        return None

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    protocol = int(packet[IP].proto)

    src_port = 0
    dst_port = 0

    if TCP in packet:
        src_port = int(packet[TCP].sport)
        dst_port = int(packet[TCP].dport)
    elif UDP in packet:
        src_port = int(packet[UDP].sport)
        dst_port = int(packet[UDP].dport)

    return (src_ip, dst_ip, src_port, dst_port, protocol)


def process_packet(packet, flows: dict):
    key = get_flow_key(packet)
    if key is None:
        return

    packet_time = float(packet.time)
    packet_len = len(packet)

    if key not in flows:
        flows[key] = {
            "src_ip": key[0],
            "dst_ip": key[1],
            "src_port": key[2],
            "dst_port": key[3],
            "protocol": key[4],
            "start_time": packet_time,
            "last_time": packet_time,
            "packets": 0,
            "bytes": 0,
            "syn_flag_count": 0,
            "ack_flag_count": 0,
            "psh_flag_count": 0,
            "urg_flag_count": 0,
        }

    flow = flows[key]
    flow["packets"] += 1
    flow["bytes"] += packet_len
    flow["last_time"] = packet_time

    if TCP in packet:
        flags = packet[TCP].flags
        if flags.S:
            flow["syn_flag_count"] += 1
        if flags.A:
            flow["ack_flag_count"] += 1
        if flags.P:
            flow["psh_flag_count"] += 1
        if flags.U:
            flow["urg_flag_count"] += 1


def build_feature_dataframe(flows: dict) -> pd.DataFrame:
    rows = []

    for flow in flows.values():
        duration = max(flow["last_time"] - flow["start_time"], 0.001)
        packets = float(flow["packets"])
        bytes_ = float(flow["bytes"])
        pps = packets / duration
        bps = bytes_ / duration
        avg_packet_size = bytes_ / max(packets, 1.0)

        protocol = int(flow["protocol"])
        is_tcp = 1 if protocol == 6 else 0
        is_udp = 1 if protocol == 17 else 0
        is_sensitive_port = 1 if flow["dst_port"] in SENSITIVE_PORTS else 0

        tcp_flag_total = (
            flow["syn_flag_count"]
            + flow["ack_flag_count"]
            + flow["psh_flag_count"]
            + flow["urg_flag_count"]
        )

        rows.append({
            "flow": f'{flow["src_ip"]}:{flow["src_port"]} -> {flow["dst_ip"]}:{flow["dst_port"]}',
            "duration": duration,
            "packets": packets,
            "bytes": bytes_,
            "pps": pps,
            "bps": bps,
            "avg_packet_size": avg_packet_size,
            "src_port": 0.0,  # keep aligned with training.py
            "dst_port": float(flow["dst_port"]),
            "is_tcp": is_tcp,
            "is_udp": is_udp,
            "is_sensitive_port": is_sensitive_port,
            "tcp_flag_total": float(tcp_flag_total),
        })

    return pd.DataFrame(rows)


def run_20_second_scan():
    flows = {}

    active_iface = detect_active_interface()
    print("Auto-detected interface:", active_iface)

    if active_iface is None:
        print("No active interface detected.")
        return [], 0, 0, 0, "No interface detected"

    def packet_handler(packet):
        process_packet(packet, flows)

    print(f"Starting 20-second scan on: {active_iface}")
    try:
        sniff(
            iface=active_iface,
            prn=packet_handler,
            store=False,
            timeout=20
        )
    except Exception as e:
        print(f"Capture error on {active_iface}: {e}")
        return [], 0, 0, 0, active_iface

    print("Scan complete.")
    print("Captured flows:", len(flows))

    df = build_feature_dataframe(flows)
    print("Feature rows:", len(df))

    if df.empty:
        return [], 0, 0, 0, active_iface

    feature_columns = [
        "duration",
        "packets",
        "bytes",
        "pps",
        "bps",
        "avg_packet_size",
        "src_port",
        "dst_port",
        "is_tcp",
        "is_udp",
        "is_sensitive_port",
        "tcp_flag_total",
    ]

    X = df[feature_columns].replace([np.inf, -np.inf], 0).fillna(0)

    preds = pipeline.predict(X)
    scores = pipeline.decision_function(X)

    results = []
    anomaly_count = 0

    for i, row in df.iterrows():
        is_anomaly = preds[i] == -1
        if is_anomaly:
            anomaly_count += 1

        results.append({
            "flow": row["flow"],
            "duration": round(float(row["duration"]), 3),
            "packets": int(row["packets"]),
            "bytes": int(row["bytes"]),
            "pps": round(float(row["pps"]), 2),
            "bps": round(float(row["bps"]), 2),
            "score": round(float(scores[i]), 4),
            "dst_port": int(row["dst_port"]),
            "protocol": "TCP" if row["is_tcp"] == 1 else ("UDP" if row["is_udp"] == 1 else "OTHER"),
            "suspicious": is_anomaly,
        })

    results.sort(key=lambda x: x["score"])

    total = len(results)
    anomalous_flows = anomaly_count
    flagged_flows = anomaly_count
    risk = round((anomaly_count / total) * 100, 2) if total else 0

    return results, anomalous_flows, flagged_flows, risk, active_iface


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/scan")
def scan():
    results, vulnerable, exploited, risk, iface_name = run_20_second_scan()
    return render_template(
        "index.html",
        flows=results,
        vulnerable=vulnerable,
        exploited=exploited,
        risk=risk,
        iface_name=iface_name
    )


def open_browser():
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Thread(target=open_browser, daemon=True).start()
    app.run(host="127.0.0.1", port=5000, debug=False, use_reloader=False)