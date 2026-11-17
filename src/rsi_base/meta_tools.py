"""Small stdio MCP transport to a token-protected host controller; no filesystem tools."""

import argparse
import json
import os
import sys
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    args = parser.parse_args()
    token = os.environ["RSI_AGENT_TOKEN"]
    for line in sys.stdin:
        request = {}
        try:
            request = json.loads(line)
            if "id" not in request:
                continue
            method, params = request["method"], request.get("params", {})
            if method == "initialize":
                result = {
                    "protocolVersion": params.get("protocolVersion", "2024-11-05"),
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "rsi-host-tools", "version": "1"},
                }
            elif method == "ping":
                result = {}
            elif method in {"tools/list", "tools/call"}:
                payload = json.dumps({"method": method, "params": params}).encode()
                message = urllib.request.Request(
                    args.url,
                    data=payload,
                    headers={
                        "Authorization": "Bearer " + token,
                        "Content-Type": "application/json",
                    },
                )
                with urllib.request.urlopen(message, timeout=300) as response:
                    result = json.load(response)
            else:
                result = {}
            reply = {"jsonrpc": "2.0", "id": request["id"], "result": result}
        except Exception as exc:
            reply = {
                "jsonrpc": "2.0",
                "id": request.get("id"),
                "error": {"code": -32603, "message": type(exc).__name__},
            }
        print(json.dumps(reply), flush=True)


if __name__ == "__main__":
    main()
