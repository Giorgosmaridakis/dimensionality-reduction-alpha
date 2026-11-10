# Walking Forward

This package contains the walk-forward experiment engine. Each file has one focused job.

- `forecaster.py`: main `WalkingForwardForecaster` class and fold loop.
- `config.py`: option validation and parameter-grid helpers.
- `data.py`: dataframe preparation and test-window offsets.
- `preprocessing.py`: final fold scaling, PCA/FA, GKX, and FactorVAE transforms.
- `tuning.py`: tree-model hyperparameter tuning bridge.
- `neural.py`: facade that combines the neural mixins.
- `neural_utils.py`: neural prediction frames, CSV writing, and validation scoring.
- `neural_tuning.py`: neural expected-return hyperparameter tuning.
- `neural_artifacts.py`: neural reuse, saved parameters, predictions, metrics, and state dicts.
- `outputs.py`: prediction files, daily IC, fold metrics, and aggregate IC output.

Import it with:

```python
from walking_forward import WalkingForwardForecaster
```
