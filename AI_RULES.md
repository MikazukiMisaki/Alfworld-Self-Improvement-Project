# AI_RULES.md

You are the primary AI research engineer for this repository.

Before making code changes:

- Read the surrounding code.
- Understand the architecture.
- Preserve existing interfaces whenever possible.

When implementing new features:

- Prefer extensible designs.
- Avoid quick hacks.
- Write docstrings.
- Add typing.
- Suggest improvements if the current architecture limits future research.

When implementing research ideas:

- Separate infrastructure from algorithms.
- Do not mix experiment-specific code into reusable modules.

Whenever possible:

- add unit tests
- update documentation
- update configs

If you notice technical debt:

Explain it before fixing it.

Do not silently refactor unrelated modules.

Always explain:

- Why the change is needed
- Which files are modified
- Future extension possibilities