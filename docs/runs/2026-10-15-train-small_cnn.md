# Run log: small_cnn (2026-10-15)

- hardware: `{"platform": "Windows-10-10.0.26200-SP0", "processor": "AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD", "logical_cpus": 16, "python": "3.11.9", "torch": "2.14.1+cpu", "device": "cpu", "torch_threads": 8}`
- config: `{"model": "small_cnn", "pretrained": false, "epochs": 15, "batch_size": 64, "lr": 0.002, "weight_decay": 0.0001, "loss": "bce", "focal_gamma": 2.0, "augment": true}`
- data: PneumoniaMNIST 128x128, official train/val/test split

| seed | best epoch | best val AUC | test AUC | seconds |
|---|---|---|---|---|
| 0 | 12 | 0.9967 | 0.9874 | 1029.7 |
| 1 | 14 | 0.9969 | 0.9839 | 586.5 |
| 2 | 12 | 0.9960 | 0.9899 | 542.3 |
| 3 | 12 | 0.9977 | 0.9875 | 445.3 |
| 4 | 11 | 0.9975 | 0.9814 | 439.3 |

Test AUC is listed for the record only; the checkpoint was chosen on validation AUC.
