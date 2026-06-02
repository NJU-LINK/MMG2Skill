---
name: git-merge
description: Merge current branch into main with --no-ff, preserving all commits for traceability
disable-model-invocation: false
argument-hint: optional commit message or instructions
---

## Context

- Current branch: !`git branch --show-current`
- Commits to merge: !`git log --oneline main..HEAD 2>/dev/null`
- Changed files: !`git diff main...HEAD --stat 2>/dev/null`
- Working tree clean: !`git status --short`

## Your task

Merge the current branch into `main` using `--no-ff`, preserving all individual commits for traceability.

### Steps

1. Ensure working tree is clean (no uncommitted changes). If not, abort and warn.
2. Recommend syncing with main first: `git rebase origin/main` (see git-sync skill). If the user declines, proceed anyway.
3. Switch to main: `git checkout main && git pull origin main`
4. Merge with no-ff: `git merge --no-ff <branch>`
   - The default merge commit message is fine, or use user-provided message
5. Confirm the merge result with `git log --oneline --graph -10`

### Rules

- **Always** use `--no-ff` to create a merge commit node, even if fast-forward is possible
- **Never** use `--squash` — individual commits must be preserved for traceability
- **Never** delete the source branch automatically — let the user decide
- If there are merge conflicts, report them and stop. Do not auto-resolve.

$ARGUMENTS
