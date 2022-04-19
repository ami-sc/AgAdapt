# GAH1 - Grain_Yield Model Summary [v0.1__XGB_LD_ATF]

***

### Model Performance

- Baseline Model [MAE] = 42.8763
- Baseline Model [RMSE] = 55.5973
- Trained Model [MAE] = 26.0480
- Trained Model [RMSE] = 33.0318
- Prediction [MAE] = 37.0611
- Prediction [RMSE] = 48.0804
***

### Dataset Statistics

- LOFO Field [Mean] = 124.6896 [bu/A]
- LOFO Field [Standard Deviation] = 46.1641 [bu/A]
- Model Dataset [Mean] = 155.8125 [bu/A]
- Model Dataset [Standard Deviation] = 45.1918 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 | 25.0578 |
|          2 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 | 25.0526 |
|          3 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|          4 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|          5 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|          6 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|          7 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|          8 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|          9 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|         10 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|         11 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|         12 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|         13 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|         14 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|         15 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|         16 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|         17 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|         18 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|         19 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |
|         20 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 | 25.0543 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 25.0596 | 25.1014 | 25.107  | 25.0601 | 25.0725 | 25.0637 | 25.0624 | 25.0574 | 25.0667 | 25.0491 |
|          0.2 | 25.0443 | 25.0384 | 25.0178 | 25.0392 | 25.0373 | 25.0267 | 25.0373 | 25.0316 | 25.0305 | 25.0185 |
|          0.3 | 25.0345 | 25.0442 | 25.0352 | 25.0443 | 25.0315 | 25.0383 | 25.0396 | 25.0333 | 25.0339 | 25.0362 |
|          0.4 | 25.0509 | 25.0543 | 25.0454 | 25.053  | 25.0493 | 25.0489 | 25.0509 | 25.0449 | 25.0445 | 25.0464 |
|          0.5 | 25.0487 | 25.0515 | 25.054  | 25.0577 | 25.046  | 25.0462 | 25.0514 | 25.0548 | 25.0459 | 25.0423 |
|          0.6 | 25.0565 | 25.0476 | 25.0469 | 25.0656 | 25.0521 | 25.0447 | 25.0522 | 25.0517 | 25.0655 | 25.054  |
|          0.7 | 25.0518 | 25.0556 | 25.0522 | 25.0584 | 25.0562 | 25.0553 | 25.0524 | 25.053  | 25.0573 | 25.0537 |
|          0.8 | 25.0553 | 25.056  | 25.0506 | 25.0521 | 25.0574 | 25.0512 | 25.0526 | 25.0525 | 25.0498 | 25.052  |
|          0.9 | 25.0504 | 25.0478 | 25.0484 | 25.047  | 25.0469 | 25.0504 | 25.0449 | 25.0501 | 25.0465 | 25.0456 |
|          1   | 25.0542 | 25.0532 | 25.0534 | 25.0543 | 25.0534 | 25.0538 | 25.0534 | 25.0543 | 25.0543 | 25.0526 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 25.0608 | 25.0388 | 25.0178 | 25.0499 | 25.0712 | 25.0675 | 59.9105 |

***

### Tuned Parameters

- max_depth = 2
- min_child_weight = 12
- subsample = 0.2
- colsample_bytree = 0.3
- eta = 0.3
- num_boost_round = 35