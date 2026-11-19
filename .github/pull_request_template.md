## Summary

Describe the technical scope and motivation for this pull request.

## Checklist

- [ ] `cargo fmt --all -- --check` passes
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` passes
- [ ] `cargo test --workspace` passes
- [ ] GPU correctness tests (`cargo test --workspace -- --ignored`) run if hardware-facing code changed
- [ ] Documentation and compatibility status claims match verified evidence
- [ ] No temporary scripts, raw dumps, generated binaries, or personal paths staged
