# Repository collaboration

- Preserve the user's configured Git author and committer identity. Use the existing repository identity for commits; do not replace it with Codex, an assistant name, or a bot email, and do not add assistant co-author trailers.
- If a commit identity is missing, establish the user's actual identity before committing. Keep identity configuration local to this repository.
- The current goal is reusable RSI infrastructure, not a novel optimization algorithm. Default demos start from an explicit simple, unoptimized harness. Lead summaries with execution correctness and improvement over that configured starting harness. Strong fixed and random-search comparisons are optional research references, not infrastructure acceptance requirements.
- Validate changes to the default starting harness or execution flow with a real Meta/Task integration run before pushing. Record the tested configuration, successes and failures. Offline checks alone do not establish runtime validation.
- Preserve raw experiment results and source hashes. Historical commit IDs corrected for attribution can be resolved through `docs/history/identity-rewrite-map.json`.
