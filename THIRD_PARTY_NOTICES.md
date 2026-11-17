# Third-party notices

- The Python dependencies listed in `requirements.txt` keep their upstream licences. They are
  installed as dependencies and are not redistributed here.
- The PINN comparator of the archived benchmark is adapted from the public DC-PINNs repository,
  https://github.com/khoshisashi/DC-PINNs , source commit `0cc9093`. The benchmark documents the
  adaptation and does not claim to reproduce every experiment of the original work.
- The Andreasen–Huge comparator of the archived benchmark uses QuantLib, which is installed by
  `optional/full_legacy_experiment.ipynb` and is not redistributed here.
- `paper/evidence/` is the output of the archived benchmark as it was produced, including a copy
  of the source modules it ran with. The same modules are under `optional/legacy_source/`.
