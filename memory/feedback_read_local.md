---
name: Always read local project directory
description: Always read from local desktop project path, never the worktree; project is now at MVP3
type: feedback
originSessionId: 8e7e83b6-8037-411d-b438-d8bef1246200
---
Always read files from the LOCAL project directory:
`C:\Users\chenp\Desktop\project\Drift-aware-Real-time-Alpha-Monitoring-and-Adaptation-System\`

Never read from the worktree path (`.claude\worktrees\...`).

**Why:** The local version is many generations ahead of git history. The worktree is stale.

**How to apply:** Any time exploring codebase or reading files, always use the local desktop path. Project is currently at MVP3.
