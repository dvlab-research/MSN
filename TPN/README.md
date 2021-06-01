## TPN

### DAVIS 2016:

- TPN (Baseline)

```
sh scripts/test_davis16_tpn.sh   # frame-by-frame propagation
```

- TPN + MSN
```
sh scripts/test_davis16_tpn_msn.sh   # MSN propagation
```

- Results:

 | Method | J_mean | J_recall | J_decay | F_mean | F_recall | F_decay |
 |:----:|:----:|:----:|:----:|:----:|:----:|:----:|
  | TPN | 0.758  |  0.904   |  0.161  | 0.739  |  0.850   |  0.145  |
 | TPN + MSN  |  **0.768**  |  **0.923**   |  **0.111**  | **0.746**  |  **0.859**   |  **0.099** | 
 

### DAVIS 2017:

- TPN (Baseline)

```
sh scripts/test_davis17_tpn.sh   # frame-by-frame propagation
```

- TPN + MSN
```
sh scripts/test_davis17_tpn_msn.sh   # MSN propagation
```

- Results:

 | Method | J_mean | J_recall | J_decay | F_mean | F_recall | F_decay |
 |:----:|:----:|:----:|:----:|:----:|:----:|:----:|
  | TPN | 0.589  |  **0.685**   |  **0.273**  | 0.627  |  0.710   |  **0.314**  |
 | TPN + MSN  |  **0.595**  |  0.675   |  0.277  | **0.633**  |  **0.714**   |  0.321 | 
 
 
### YTVOS:

- TPN (Baseline)

```
sh scripts/test_ytvos_tpn.sh   # frame-by-frame propagation
```

- TPN + MSN
```
sh scripts/test_ytvos_tpn_msn.sh   # MSN propagation
```

- Results:

 | Method | J_seen | J_unseen | F_seen | F_unseen | Overall |
 |:----:|:----:|:----:|:----:|:----:|:----:|
 | TPN   | 0.640  |  0.570   |  0.659  | 0.654 |  0.631   |
 | TPN + MSN  |  **0.657**  |  **0.581**   |  **0.680**  | **0.664**  |  **0.645**   |
 
