# TXH2 - Grain_Yield Model Summary [v0.2__XGB_PCA_ATF]

***

### Model Performance

- Baseline Model [MAE] = 33.4986
- Baseline Model [RMSE] = 40.4976
- Trained Model [MAE] = 25.6676
- Trained Model [RMSE] = 32.9568
- Prediction [MAE] = 48.7245
- Prediction [RMSE] = 58.1997
***

### Dataset Statistics

- LOFO Field [Mean] = 133.6854 [bu/A]
- LOFO Field [Standard Deviation] = 34.1392 [bu/A]
- Model Dataset [Mean] = 155.5735 [bu/A]
- Model Dataset [Standard Deviation] = 45.6826 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  | 25.533  |
|          2 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|          3 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|          4 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|          5 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|          6 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|          7 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|          8 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|          9 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|         10 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|         11 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|         12 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|         13 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|         14 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|         15 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|         16 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|         17 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|         18 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|         19 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |
|         20 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 25.5338 | 25.5452 | 25.5548 | 25.5376 | 25.5421 | 25.5409 | 25.5503 | 25.6064 | 25.6072 | 25.5456 |
|          0.2 | 25.5469 | 25.5558 | 25.5386 | 25.5251 | 25.5305 | 25.531  | 25.5265 | 25.5192 | 25.5329 | 25.5576 |
|          0.3 | 25.5478 | 25.5007 | 25.5193 | 25.5492 | 25.5158 | 25.5499 | 25.5466 | 25.5512 | 25.5504 | 25.5485 |
|          0.4 | 25.5034 | 25.547  | 25.5333 | 25.5063 | 25.5384 | 25.5083 | 25.5081 | 25.5057 | 25.5012 | 25.5373 |
|          0.5 | 25.5085 | 25.5425 | 25.5376 | 25.5417 | 25.5184 | 25.5119 | 25.5446 | 25.5447 | 25.5421 | 25.5443 |
|          0.6 | 25.5154 | 25.5363 | 25.5162 | 25.5216 | 25.5234 | 25.5137 | 25.5172 | 25.5188 | 25.5225 | 25.522  |
|          0.7 | 25.522  | 25.5307 | 25.5311 | 25.5231 | 25.5211 | 25.5233 | 25.5215 | 25.5365 | 25.5373 | 25.5374 |
|          0.8 | 25.5266 | 25.5333 | 25.5263 | 25.534  | 25.5323 | 25.5248 | 25.5301 | 25.5252 | 25.5318 | 25.5258 |
|          0.9 | 25.5318 | 25.5313 | 25.5294 | 25.5341 | 25.5275 | 25.5293 | 25.529  | 25.5302 | 25.5276 | 25.529  |
|          1   | 25.5305 | 25.5303 | 25.5305 | 25.5305 | 25.53   | 25.5305 | 25.5305 | 25.5305 | 25.5305 | 25.5305 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|---------|---------|---------|---------|---------|---------|
| MAE   | 25.5351 | 25.5288 | 25.5007 | 25.5184 | 25.5492 | 25.5421 | 59.4973 |

***

### Tuned Parameters

- max_depth = 3
- min_child_weight = 4
- subsample = 0.3
- colsample_bytree = 0.2
- eta = 0.3
- num_boost_round = 42