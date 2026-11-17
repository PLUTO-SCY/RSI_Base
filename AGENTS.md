# Repository collaboration

- Preserve the user's configured Git author and committer identity. Use the existing repository identity for commits; do not replace it with Codex, an assistant name, or a bot email, and do not add assistant co-author trailers.
- If a commit identity is missing, establish the user's actual identity before committing. Keep identity configuration local to this repository.
- The goal is reusable RSI infrastructure developed incrementally. Use initial/default harness terminology.
- Validate changes to the default configuration or execution flow with a real Meta/Task integration run before pushing. Keep tested commands, configuration, successes and failures in ignored local output; offline checks alone do not establish runtime validation.
- Keep the current repository tree focused on reusable source, scripts, configuration, dependency/data locks, tests and necessary usage documentation. Do not commit experiment iteration records, generated reports/figures, historical candidates, raw logs, datasets, model weights, environments or secrets.
- Preserve raw experiment results and their source hashes locally. Removing old material from the current tree does not authorize rewriting Git history.
