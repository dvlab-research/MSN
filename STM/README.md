## STM

### DAVIS16:

- STM-1 (Baseline)

```
sh scripts/test_baseline_davis16.sh   # frame-by-frame propagation
```

- STM-1 + MSN

```
sh scripts/select_davis16.sh     # Generate selection file 
sh scripts/test_msn_davis16.sh   # MSN propagation
```

- Results:

 | Method | J_mean | J_recall | J_decay | F_mean | F_recall | F_decay |
 |:----:|:----:|:----:|:----:|:----:|:----:|:----:|
  | STM-1 | 0.832  |  **0.928**   |  0.151  | 0.833  |  0.902   |  0.143  |
 | STM-1 + MSN  |  **0.838**  |  0.925   |  **0.118**  | **0.849**  |  **0.913**   |  **0.126** | 
 
### DAVIS 2017:

- STM-1 (Baseline)

```
sh scripts/test_baseline_davis17.sh   # frame-by-frame propagation
```

- STM-1 + MSN

```
sh scripts/select_davis17.sh     # Generate selection file 
sh scripts/test_msn_davis17.sh   # MSN propagation
```
  
- Results:

 | Method | J_mean | J_recall | J_decay | F_mean | F_recall | F_decay |
 |:----:|:----:|:----:|:----:|:----:|:----:|:----:|
  | STM-1   | 0.696  |  0.794   |  **0.204**  | 0.746  |  0.828   |  0.234 |
 | STM-1 + MSN | **0.714**  |  **0.802**   |  0.211  | **0.768**  |  **0.851**   |  **0.229** |


### YTVOS:

- STM-1 (Baseline)

```
sh scripts/test_baseline_ytvos.sh   # frame-by-frame propagation
```

- STM-1 + MSN
```
sh scripts/select_ytvos.sh     # Generate selection file 
sh scripts/test_msn_ytvos.sh   # MSN propagation
```

- Results:

 | Method | J_seen | J_unseen | F_seen | F_unseen | Overall |
 |:----:|:----:|:----:|:----:|:----:|:----:|
 | STM-1   | 0.717  |  0.640   |  0.744  | 0.697  |  0.699   |
 | STM-1 + MSN  |  **0.724**  |  **0.654**   |  **0.752**  | **0.714**  |  **0.711**   |

  

