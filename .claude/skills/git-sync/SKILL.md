---
name: git-sync
description: Rebase current branch onto main to sync latest changes
disable-model-invocation: false
argument-hint: optional target branch (defaults to main)
---

## Context

- Current branch: !`git branch --show-current`
- Working tree clean: !`git status --short`
- Commits ahead of main: !`git log --oneline main..HEAD 2>/dev/null`
- Commits behind main: !`git log --oneline HEAD..origin/main 2>/dev/null`

## Your task

Sync the current feature branch with the latest main using rebase, keeping history linear.

### Steps

1. Ensure working tree is clean. If not, abort and warn.
2. Fetch latest: `git fetch origin`
3. Rebase onto main: `git rebase origin/main`
4. If conflicts occur during rebase, report the conflicting files and stop. Do not auto-resolve. Inform the user they can abort with `git rebase --abort`.
5. If the branch was previously pushed, inform the user they need to force push: `git push --force-with-lease`
6. Confirm result with `git log --oneline --graph -5`

### Rules

- **Always** use rebase (not merge) to sync from main — keeps feature branch history linear
- **Never** run `git push --force-with-lease` without informing the user first
- If the branch is shared with others (multiple collaborators), warn that rebase rewrites history and suggest `git merge main` as a safer alternative

$ARGUMENTS
