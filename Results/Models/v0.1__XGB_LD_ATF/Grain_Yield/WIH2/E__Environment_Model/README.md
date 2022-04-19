# WIH2 - Grain_Yield Model Summary [v0.1__XGB_LD_ATF]

***

### Model Performance

- Baseline Model [MAE] = 37.2996
- Baseline Model [RMSE] = 43.9803
- Trained Model [MAE] = 26.0183
- Trained Model [RMSE] = 33.2323
- Prediction [MAE] = 47.6916
- Prediction [RMSE] = 54.3826
***

### Dataset Statistics

- LOFO Field [Mean] = 183.3416 [bu/A]
- LOFO Field [Standard Deviation] = 31.5944 [bu/A]
- Model Dataset [Mean] = 152.7188 [bu/A]
- Model Dataset [Standard Deviation] = 45.6962 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 | 25.6363 |
|          2 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 | 25.6365 |
|          3 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|          4 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|          5 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|          6 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|          7 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|          8 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|          9 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|         10 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|         11 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|         12 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|         13 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|         14 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|         15 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|         16 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|         17 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|         18 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|         19 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |
|         20 | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  | 25.637  |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 25.764  | 25.755  | 25.8183 | 25.8187 | 25.7218 | 25.7622 | 25.7143 | 25.7775 | 25.7643 | 25.7581 |
|          0.2 | 25.7346 | 25.6676 | 25.6831 | 25.6697 | 25.6598 | 25.6344 | 25.6245 | 25.646  | 25.6404 | 25.6443 |
|          0.3 | 25.6066 | 25.6464 | 25.6569 | 25.6548 | 25.6874 | 25.65   | 25.6493 | 25.6484 | 25.6921 | 25.6571 |
|          0.4 | 25.6653 | 25.6468 | 25.6365 | 25.6824 | 25.6999 | 25.6675 | 25.6868 | 25.6657 | 25.6686 | 25.6611 |
|          0.5 | 25.6271 | 25.6153 | 25.6308 | 25.6332 | 25.6694 | 25.6206 | 25.6673 | 25.6807 | 25.666  | 25.6776 |
|          0.6 | 25.6355 | 25.638  | 25.6848 | 25.6151 | 25.6149 | 25.6323 | 25.6316 | 25.685  | 25.6836 | 25.6247 |
|          0.7 | 25.6298 | 25.6551 | 25.6272 | 25.6158 | 25.6332 | 25.6287 | 25.6229 | 25.6262 | 25.6418 | 25.6217 |
|          0.8 | 25.6387 | 25.6293 | 25.631  | 25.6357 | 25.6314 | 25.6435 | 25.6352 | 25.6287 | 25.6277 | 25.6354 |
|          0.9 | 25.6452 | 25.6353 | 25.6547 | 25.6455 | 25.6359 | 25.6474 | 25.6432 | 25.6312 | 25.6332 | 25.6438 |
|          1   | 25.6369 | 25.636  | 25.6369 | 25.6367 | 25.6371 | 25.6375 | 25.6398 | 25.6371 | 25.6415 | 25.6363 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 25.6482 | 25.6571 | 25.6066 | 25.6589 | 25.6977 | 26.7074 | 60.6418 |

***

### Tuned Parameters

- max_depth = 1
- min_child_weight = 1
- subsample = 0.3
- colsample_bytree = 0.1
- eta = 0.3
- num_boost_round = 91