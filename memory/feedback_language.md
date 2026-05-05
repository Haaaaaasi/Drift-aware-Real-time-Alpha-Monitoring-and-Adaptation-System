---
name: Traditional Chinese for all docs
description: Project requires Traditional Chinese for all written documentation; English identifiers only in code
type: feedback
originSessionId: c4dcea44-dc00-4c6d-ad4f-c06e333e2afc
---
All project documentation — CLAUDE.md, docs/, notebooks/ narrative cells, reports/*.md, commit message bodies — must be written in Traditional Chinese (繁體中文). Code-level identifiers (function names, variables, class names) stay English; only comments and string-content docs are Chinese.

**Why:** Stated explicitly at the top of CLAUDE.md. This is a research project for graduate school application where the reviewer reads Traditional Chinese.

**How to apply:** When creating any new .md file, notebook narrative, or long-form output, write in Traditional Chinese. Keep variable names, pytest test names, and log message keys in English. Matplotlib titles in Chinese are fine even though DejaVu Sans emits glyph warnings (non-blocking).
