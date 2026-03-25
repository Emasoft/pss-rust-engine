# Rechecker: Merge Pending

The rechecker plugin reviewed your latest commit and fixed bugs.

- **Worktree**: `rck-64eaa6`
- **Branch with fixes**: `worktree-rck-64eaa6`

## What you must do

**Read the full report** at the path above, then merge all rechecker fixes at once:

```bash
cd "/Users/emanuelesabetta/Code/PERFECT_SKILL_SUGGESTER/perfect-skill-suggester/rust" && bash .rechecker/merge-worktrees.sh
```

The merge script handles ancestry checks, conflict resolution, worktree cleanup,
branch deletion, and report archival. Do NOT merge manually with `git merge`.
