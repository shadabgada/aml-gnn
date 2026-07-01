# Code Flow Reference

## Call chain per runner

### run_baselines.py

```
run_baselines.py
  |
  ├── loader.load_raw_data(cfg)
  ├── graph_constructor.build_static_graph(accounts, txns, cfg)
  |     └── inside: feature_engineering.build_node_features() / build_edge_features()
  |
  ├── baselines.LogisticRegression / RandomForest / XGBoost
  ├── evaluator.evaluate_baseline(model, X_train, y_train, ...)
  |     └── calls metrics.compute_all_metrics()
  ├── config.DataConfig
  └── logger.setup_logging()
```

### run_gnn.py

```
run_gnn.py
  |
  ├── loader.load_raw_data(cfg)
  ├── graph_constructor.build_static_graph(accounts, txns, cfg)
  |
  ├── gcn.GCNModel / gat.GATModel / sage.SAGEModel
  ├── trainer.train_model(model, data, cfg)
  |     └── calls model(data)           [forward pass on full graph]
  |     └── calls metrics.compute_all_metrics()  [validation]
  |     └── calls metrics.calibrate_threshold()  [best F1 threshold]
  |
  ├── evaluator.evaluate_model(model, data)
  |     └── calls model.predict(data)   [no grad]
  |     └── calls metrics.compute_all_metrics()
  |
  ├── config.DataConfig, ModelConfig, TrainingConfig
  └── logger.setup_logging()
```

### run_temporal.py

```
run_temporal.py
  |
  ├── loader.load_raw_data(cfg)
  ├── graph_constructor.build_temporal_snapshots(accounts, txns, cfg)
  |     └── returns list of 12 PyG Data objects (snapshots)
  |
  ├── temporal_gnn.TemporalGCN / EvolveGCNH
  ├── temporal_trainer.train_temporal(model, snapshots, cfg)
  |     └── loops over snapshots, calls model(snapshot, prev_state)
  |     └── GRU evolves state across snapshots (TemporalGCN) or weights (EvolveGCNH)
  |
  ├── evaluator.evaluate_model(model, test_snapshots)
  ├── config
  └── logger
```

### run_tgn.py

```
run_tgn.py
  |
  ├── loader.load_raw_data(cfg)
  ├── temporal_data_builder.build_temporal_data(accounts, txns, cfg)
  |     └── returns TemporalData (continuous-time edge stream, no snapshots)
  |
  ├── tgn_model.TGNModel(node_dim, edge_dim, memory_dim=64, time_dim=8)
  |     ├── inside: EMAMemory(beta=0.85)          ← per-node memory store
  |     ├── inside: TimeEncoder(time_dim)          ← sinusoidal time features
  |     ├── inside: NodeProjection(node_dim -> hidden_dim)
  |     ├── inside: MessageProjection(concat -> memory_dim)
  |     └── inside: EdgeClassifier(concat -> 1)
  |
  ├── tgn_trainer.train_tgn(model, temporal_data, cfg)
  |     └── processes edges in batches, chronologically ordered
  |     └── for each batch:
  |           1. model.forward(data)    → predict with OLD memory
  |           2. model.update_memory()  → EMA: m_new = beta*m_old + (1-beta)*msg
  |     └── calls metrics.calibrate_threshold()
  |
  ├── evaluator.evaluate_model(model, temporal_data)
  |     └── model.predict(data)  → also uses old memory, no updates
  |     └── per-slice analysis across 12 test time windows
  |
  ├── config
  └── logger
```

## Cross-file: who calls whom

| File | Defines | Called by |
|------|---------|-----------|
| `loader.py` | `load_raw_data()` | all 4 runners |
| `feature_engineering.py` | `build_node_features()`, `build_edge_features()` | `graph_constructor.py` (internal) |
| `graph_constructor.py` | `build_static_graph()`, `build_temporal_snapshots()` | `run_baselines.py`, `run_gnn.py`, `run_temporal.py` |
| `temporal_data_builder.py` | `build_temporal_data()` | `run_tgn.py` |
| `gcn.py` | `GCNModel` | `run_gnn.py`, `trainer.py`, `evaluator.py` |
| `gat.py` | `GATModel` | `run_gnn.py`, `trainer.py`, `evaluator.py` |
| `sage.py` | `SAGEModel` | `run_gnn.py`, `trainer.py`, `evaluator.py` |
| `temporal_gnn.py` | `TemporalGCN`, `EvolveGCNH` | `run_temporal.py`, `temporal_trainer.py` |
| `tgn_model.py` | `TGNModel`, `EMAMemory`, `TimeEncoder`, `NodeProjection`, `MessageProjection`, `EdgeClassifier` | `run_tgn.py`, `tgn_trainer.py` |
| `baselines.py` | `LogisticRegression`, `RandomForest`, `XGBoost` | `run_baselines.py` |
| `trainer.py` | `train_model()` | `run_gnn.py` |
| `temporal_trainer.py` | `train_temporal()` | `run_temporal.py` |
| `tgn_trainer.py` | `train_tgn()` | `run_tgn.py` |
| `evaluator.py` | `evaluate_model()`, `evaluate_baseline()` | all 4 runners |
| `metrics.py` | `compute_all_metrics()`, `calibrate_threshold()`, `format_metrics()` | `trainer.py`, `temporal_trainer.py`, `tgn_trainer.py`, `evaluator.py` |
| `config.py` | `DataConfig`, `ModelConfig`, `TrainingConfig` | all runners, all trainers |
| `logger.py` | `setup_logging()` | all runners |
