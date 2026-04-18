# Host adapter for Antigravity

Before work, read `./CLAUDE.md` and treat it as the authoritative project brief.

Do not apply `./CLAUDE.local.md` automatically. That file is specific to Claude Code running inside a rootless Podman container and does not describe the default Antigravity environment.

Assume Antigravity is running on the host:
- Use the actual repository root as the working directory.
- Prefer repo-relative paths over `/workspace/...`.
- Host tools and host paths may be available if they are in the normal project environment.
- Do not assume Claude container paths such as `/workspace` or `/home/node/.claude`.

Preserve all project-level rules from `CLAUDE.md`, especially:
- research vs testing mode separation
- scope discipline
- file-reading discipline
- benchmark analysis order
- queue and handoff protocol

Only consult `CLAUDE.local.md` if the user explicitly asks for Claude-container compatibility, wrapper commands, or path translation between host and container.
