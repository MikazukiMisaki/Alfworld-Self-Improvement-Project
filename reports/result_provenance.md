# Historical Result Provenance

Classification uses artifact contents first. Directory names are only supplementary evidence.

- Runs: 15
- Pipelines: `{"free_form_v1": 13, "indexed_v1": 2}`
- Artifact types: `{"diagnostic": 1, "historical_run": 5, "smoke_test": 9}`
- Verified successful legacy cases: 0

| Run | Pipeline | Mode | Type | Episodes | Success | Confidence |
|---|---|---|---|---:|---:|---|
| `20260809T112835Z` | `free_form_v1` | `free_form_validated` | `historical_run` | 1 | 0.000 | medium |
| `smoke-a40` | `free_form_v1` | `free_form_validated` | `smoke_test` | 1 | unknown | medium |
| `smoke-a40-001` | `free_form_v1` | `free_form_validated` | `smoke_test` | 1 | unknown | medium |
| `smoke-a40-002` | `free_form_v1` | `free_form_validated` | `smoke_test` | 1 | unknown | medium |
| `smoke-a40-003` | `free_form_v1` | `free_form_validated` | `smoke_test` | 1 | unknown | medium |
| `smoke-a40-004` | `free_form_v1` | `free_form_validated` | `smoke_test` | 1 | 0.000 | medium |
| `smoke-a40-005` | `free_form_v1` | `free_form_validated` | `smoke_test` | 1 | 0.000 | medium |
| `smoke-a40-006` | `free_form_v1` | `free_form_validated` | `smoke_test` | 1 | 0.000 | medium |
| `smoke-a40-010` | `free_form_v1` | `free_form_validated` | `smoke_test` | 1 | unknown | medium |
| `smoke-a40-011` | `free_form_v1` | `free_form_validated` | `smoke_test` | 1 | 0.000 | medium |
| `sprint1-5-acc28b43e0d2-20260809T114656Z` | `free_form_v1` | `free_form_validated` | `historical_run` | 1 | 0.000 | medium |
| `sprint1-5-acc28b43e0d2-20260813T162958Z` | `indexed_v1` | `indexed_admissible` | `historical_run` | 1 | 0.000 | high |
| `sprint1-5-b1-diagnostic-20260813T163453Z` | `indexed_v1` | `indexed_admissible` | `diagnostic` | 3 | 0.000 | high |
| `sprint1-5-sdxsrv03-20260809T112117Z` | `free_form_v1` | `free_form_validated` | `historical_run` | 1 | 0.000 | medium |
| `sprint1-5-sdxsrv03-20260809T112909Z` | `free_form_v1` | `free_form_validated` | `historical_run` | 1 | 0.000 | medium |

## Limitations

The historical `AlfWorldLegacy` runner and its reported 210-episode/28-success corpus are not available in this checkout or in the audited server workspace. None of the available tracked trajectories succeeded, so no legacy-success regression case can be verified from current evidence.

Historical manifests do not contain a complete dependency, dataset, and hardware fingerprint. Their runtime environments remain only partially attributable even when pipeline classification is strong.
