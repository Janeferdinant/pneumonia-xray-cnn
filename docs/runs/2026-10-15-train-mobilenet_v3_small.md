# Run log: mobilenet_v3_small (2026-10-15)

- hardware: `{"platform": "Windows-10-10.0.26200-SP0", "processor": "AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD", "logical_cpus": 16, "python": "3.11.9", "torch": "2.14.1+cpu", "device": "cpu", "torch_threads": 8}`
- config: `{"model": "mobilenet_v3_small", "pretrained": true, "epochs": 10, "batch_size": 64, "lr": 0.0005, "weight_decay": 0.0001, "loss": "bce", "focal_gamma": 2.0, "augment": true}`
- data: PneumoniaMNIST 128x128, official train/val/test split

| seed | best epoch | best val AUC | test AUC | seconds |
|---|---|---|---|---|
| 0 | 8 | 0.9989 | 0.9893 | 362.6 |
| 1 | 10 | 0.9994 | 0.9885 | 335.6 |
| 2 | 10 | 0.9991 | 0.9923 | 347.2 |
| 3 | 10 | 0.9986 | 0.9878 | 341.9 |
| 4 | 9 | 0.9993 | 0.9910 | 369.8 |

Test AUC is listed for the record only; the checkpoint was chosen on validation AUC.
