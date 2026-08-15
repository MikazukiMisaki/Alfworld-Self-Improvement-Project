# Experiment Artifacts

`artifacts/runtime/` is the default destination for new experiment output and
is ignored by Git. A runtime run is evidence only when its manifest, Git
revision, resolved configuration, environment, and trajectory files have been
validated.

Do not place test fixtures here. Small synthetic fixtures belong under
`tests/fixtures/`. Historical artifacts already committed under `results/`
remain an immutable archive and are not valid implicit inputs to new runs.
