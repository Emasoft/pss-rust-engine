# PSS Rust Engine

Rust source code for the [Perfect Skill Suggester](https://github.com/Emasoft/perfect-skill-suggester) Claude Code plugin.

This repo is used as a **git submodule** by the main PSS plugin. Pre-built binaries are committed to the plugin repo — you only need this source to build from scratch or contribute.

## Crates

| Crate | Binary | Description |
|-------|--------|-------------|
| `skill-suggester` | `pss` | Multi-signal scorer for skill/agent/command suggestions |
| `negation-detector` | `pss-nlp` | NLP-based negation detection (POS tagging, chunking) |

## Building

```bash
# Build both binaries (release mode)
cargo build --release

# Binaries at:
# target/release/pss
# target/release/pss-nlp
```

## Cross-compilation

See the main PSS repo's `scripts/pss_build.py` for cross-compilation targets (darwin-arm64, darwin-x86_64, linux-arm64, linux-x86_64, windows-x86_64).
