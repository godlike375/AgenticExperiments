from __future__ import annotations

import argparse
import json
import os
import sys


def _main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True)
    parser.add_argument("--tools-dir", required=True)
    parser.add_argument("--project-root", default="")
    opts = parser.parse_args()

    if opts.project_root:
        from universal_agents.project_root import set_project_root
        set_project_root(opts.project_root)

    from universal_agents import tool_session
    from universal_agents.tool_registry import load_external_plugins
    from universal_agents.tool_worker import invoke_tool

    group_tools = {
        n: f
        for n, f in load_external_plugins(opts.tools_dir).items()
        if getattr(f, "_tool_group", "internal") == opts.group
    }
    sys.stderr.write(f"[worker {opts.group}] ready, tools={sorted(group_tools)}\n")
    sys.stderr.flush()

    while True:
        line = sys.stdin.readline()
        if not line:
            break
        try:
            req = json.loads(line)
        except json.JSONDecodeError:
            break
        session = req.get("session") or {}
        token = tool_session.set_session(session)
        try:
            func = group_tools.get(req["name"])
            if func is None:
                resp = {"status": "error",
                        "value": f"tool '{req['name']}' not available in group '{opts.group}'",
                        "session": session}
            else:
                status, payload = invoke_tool(func, req.get("args", {}), req.get("dry_run", False))
                resp = {"status": status, "value": payload, "session": session}
        except Exception as e:
            resp = {"status": "error", "value": f"{type(e).__name__}: {e}", "session": session}
        finally:
            tool_session.reset_session(token)
        sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    _main()