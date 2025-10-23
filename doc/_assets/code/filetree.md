    └── examples/
        ├── demo/
        │   ├── alarm
        │   ├── move
        │   ├── ...
        │   └── <model-name>/
        │       ├── 000-<model-name>.tree
        │       └── 000-<model-name>.parquet
        ├── move/
        ├── ...
        ├── <model-name>/
        │   ├── <YYYY-MM-DD_HH:mm>/
        │   │   ├── [crossval/] (optional)
        │   │   ├── data/
        │   │   │   └── 000-<model-name>.parquet
        │   │   ├── plots/
        │   │   │   ├── 000-<model-name>.svg
        │   │   │   ├── 000-<model-name>-nodist.svg
        │   │   │   └── ...
        │   │   └── 000-<model-name>.tree
        │   ├── ...
        │   ├── __init__.py
        │   └── <model-name>.py
        └── ...