# Run log: small_cnn_unweighted (2026-10-15)

- hardware: `{"platform": "Windows-10-10.0.26200-SP0", "processor": "AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD", "logical_cpus": 16, "python": "3.11.9", "torch": "2.14.1+cpu", "device": "cpu", "torch_threads": 8}`
- config: `{"model": "small_cnn", "pretrained": false, "epochs": 15, "batch_size": 64, "lr": 0.002, "weight_decay": 0.0001, "loss": "bce_unweighted", "focal_gamma": 2.0, "augment": true}`
- data: PneumoniaMNIST 128x128, official train/val/test split

| seed | best epoch | best val AUC | test AUC | seconds |
|---|---|---|---|---|
| 0 | 13 | 0.9981 | 0.9885 | 402.4 |
| 1 | 14 | 0.9971 | 0.9818 | 399.1 |
| 2 | 10 | 0.9979 | 0.9750 | 403.8 |

Test AUC is listed for the record only; the checkpoint was chosen on validation AUC.
