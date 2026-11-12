<!-- ai-instructions: managed; source-sha256=d41e365c41f4310289c1768be5637936c56b6c1c3bd8a64667039bf4bdb1a423; scope=project; source=projects/multispecies_fuelcycle/instructions.md -->
<!-- Edit the canonical source in ~/.agents; run ai-instructions sync. -->

# Multispecies fuelcycle project instructions

## Environment and gate

- Use the `ddstartupenv` conda environment and run `python -m pytest` from the
  repository root; pytest options live in `pyproject.toml`.
- The distribution is also named `ddstartup`. Tests import the local `src.*`
  through `tests/conftest.py`, but the editable `ddstartup` package in
  `ddstartupenv` points at the `dd_startup` checkout, so a bare
  `import ddstartup` loads the other project.

## Scientific scope

This file makes no claim about physics or numerical results. Validate such
claims with this project's own tests, methods, and scientific review; do not
assume validity from DDstartup or infer it from code similarity or names.
