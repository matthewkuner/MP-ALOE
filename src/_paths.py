from pathlib import Path

_root = Path(__file__).resolve().parent / ".."
data_path = _root / "data"
models_path = data_path / "models"
candidates_path = _root / "candidate_structures_for_dft"
