# Legacy Materials

This directory is reserved for immutable, provenance-labelled imports from the
older implementation. Sprint 1 does not import, execute, or train on legacy
artifacts.

The legacy runner referenced by the research notes,
`AlfWorldLegacy/src/envs/batch_run_alfworld.py`, and its claimed 210-episode
corpus are not present in this checkout or elsewhere in the audited server
workspace as of the 2026-08-15 audit. No legacy success case can therefore be verified or
imported from the available files. Do not reconstruct or fabricate those cases
from aggregate counts in planning documents.

## Future import rule

Recovered legacy trajectories must remain immutable. Do not rewrite them to
match the current manifest schema. A future verified import must instead add a
sidecar provenance record containing source hashes, the original location and
revision when available, task ID, split, seed, and legacy pipeline identity.
The importer is intentionally not implemented during Sprint 1.5.
