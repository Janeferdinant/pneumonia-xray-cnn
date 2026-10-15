# Run log: mobilenet_v3_small_scratch (2026-10-15)

- hardware: `{"platform": "Windows-10-10.0.26200-SP0", "processor": "AMD64 Family 25 Model 33 Stepping 0, AuthenticAMD", "logical_cpus": 16, "python": "3.11.9", "torch": "2.14.1+cpu", "device": "cpu", "torch_threads": 8}`
- config: `{"model": "mobilenet_v3_small", "pretrained": false, "epochs": 10, "batch_size": 64, "lr": 0.0005, "weight_decay": 0.0001, "loss": "bce", "focal_gamma": 2.0, "augment": true}`
- data: PneumoniaMNIST 128x128, official train/val/test split

| seed | best epoch | best val AUC | test AUC | seconds |
|---|---|---|---|---|
| 0 | 10 | 0.9954 | 0.9722 | 322.4 |
| 1 | 10 | 0.9959 | 0.9687 | 310.8 |
| 2 | 10 | 0.9943 | 0.9690 | 302.8 |

Test AUC is listed for the record only; the checkpoint was chosen on validation AUC.
