#!/usr/bin/env python3
"""
Quick validation script - run this after `clab deploy` to confirm
BGP is up and the routing table is populated.

Usage:
    pip install requests
    python3 verify_lab.py
"""

import json
import requests
from requests.auth import HTTPBasicAuth

# containerlab assigns predictable management IPs
NODES = {
    "spine1": "172.20.20.2",
    "leaf1":  "172.20.20.3",
}

EAPI_USER = "admin"
EAPI_PASS = ""  # cEOS default: no password

def eapi_call(host, commands):
    url = f"https://{host}/command-api"
    payload = {
        "jsonrpc": "2.0",
        "method": "runCmds",
        "params": {"version": 1, "cmds": commands, "format": "json"},
        "id": 1,
    }
    r = requests.post(url, json=payload,
                      auth=HTTPBasicAuth(EAPI_USER, EAPI_PASS),
                      verify=False, timeout=10)
    return r.json()["result"]

def check_bgp(host):
    result = eapi_call(host, ["show bgp summary"])
    peers = result[0]["vrfs"]["default"]["peers"]
    for peer_ip, info in peers.items():
        state = info.get("peerState", "unknown")
        print(f"  [{host}] BGP peer {peer_ip}: {state}")
        if state != "Established":
            print(f"  ⚠️  Not established yet — wait 30s and retry")

def check_routes(host):
    result = eapi_call(host, ["show ip route"])
    routes = result[0]["vrfs"]["default"]["routes"]
    print(f"  [{host}] Routes in table: {len(routes)}")
    for prefix in routes:
        print(f"    {prefix}")

if __name__ == "__main__":
    import urllib3
    urllib3.disable_warnings()

    print("=== BGP Status ===")
    for name in NODES:
        check_bgp(NODES[name])

    print("\n=== Routing Tables ===")
    for name in NODES:
        check_routes(NODES[name])
