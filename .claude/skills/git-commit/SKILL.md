---
name: git-commit
description: Create a git commit using Conventional Commits format
disable-model-invocation: false
argument-hint: optional message or instructions
---

## Context

- Current git status: !`git status`
- Current git diff (staged and unstaged changes): !`git diff HEAD`
- Current branch: !`git branch --show-current`
- Recent commits: !`git log --oneline -5 2>/dev/null`

## Your task

Based on the above changes, create a single git commit using **Conventional Commits** format.

### Commit message format

```
<type>(<scope>): <short description>

<body>
```

### Type must be one of:

- **feat**: A new feature
- **fix**: A bug fix
- **docs**: Documentation only changes
- **style**: Changes that do not affect the meaning of the code (white-space, formatting, etc.)
- **refactor**: A code change that neither fixes a bug nor adds a feature
- **perf**: A code change that improves performance
- **test**: Adding missing tests or correcting existing tests
- **build**: Changes that affect the build system or external dependencies
- **ci**: Changes to CI configuration files and scripts
- **chore**: Other changes that don't modify src or test files
- **revert**: Reverts a previous commit

### Rules

- `<scope>` is optional, describes the module or area affected (e.g. `auth`, `api`, `ui`)
- `<short description>` must be lowercase, imperative mood, no period at end, max 72 chars
- `<body>` is optional, use it only when the "why" is not obvious from the description
- If there are **BREAKING CHANGES**, add `!` after type/scope: `feat(api)!: remove legacy endpoint`
- Always pass the commit message via a HEREDOC for correct formatting

You have the capability to call multiple tools in a single response. Stage and create the commit using a single message. Do not use any other tools or do anything else. Do not send any other text or messages besides these tool calls.

$ARGUMENTS
