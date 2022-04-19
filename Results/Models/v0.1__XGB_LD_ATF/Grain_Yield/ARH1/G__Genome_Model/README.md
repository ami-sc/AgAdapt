# ARH1 - Grain_Yield Model Summary [v0.1__XGB_LD_ATF]

***

### Model Performance

- Baseline Model [MAE] = 61.4648
- Baseline Model [RMSE] = 71.8291
- Trained Model [MAE] = 34.0088
- Trained Model [RMSE] = 43.0960
- Prediction [MAE] = 63.6306
- Prediction [RMSE] = 74.4091
***

### Dataset Statistics

- LOFO Field [Mean] = 98.0852 [bu/A]
- LOFO Field [Standard Deviation] = 40.6363 [bu/A]
- Model Dataset [Mean] = 157.3550 [bu/A]
- Model Dataset [Standard Deviation] = 44.1013 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 34.1293 | 34.1293 | 34.1293 | 34.2032 | 34.2032 | 34.2032 | 34.2032 | 34.2032 | 34.2032 | 34.1091 | 34.1409 | 34.1286 | 34.1286 | 34.1286 | 34.1286 | 34.1286 | 34.1286 | 34.1286 | 34.1286 | 34.251  | 34.251  |
|          2 | 33.4981 | 33.4981 | 33.4701 | 33.4827 | 33.5071 | 33.4697 | 33.4028 | 33.5157 | 33.4305 | 33.4594 | 33.5316 | 33.5824 | 33.5313 | 33.5956 | 33.535  | 33.5515 | 33.6581 | 33.6214 | 33.6631 | 33.5694 | 33.5496 |
|          3 | 33.43   | 33.43   | 33.4547 | 33.4212 | 33.4668 | 33.4526 | 33.4178 | 33.4373 | 33.4391 | 33.4117 | 33.4178 | 33.4012 | 33.5655 | 33.4808 | 33.5315 | 33.5161 | 33.5158 | 33.5133 | 33.545  | 33.4907 | 33.5321 |
|          4 | 33.4071 | 33.4071 | 33.4005 | 33.4368 | 33.4484 | 33.3979 | 33.3726 | 33.4089 | 33.4394 | 33.4176 | 33.396  | 33.4199 | 33.4658 | 33.4969 | 33.5095 | 33.4995 | 33.4987 | 33.4701 | 33.4363 | 33.5171 | 33.4551 |
|          5 | 33.3682 | 33.3682 | 33.3546 | 33.3738 | 33.3766 | 33.3807 | 33.3931 | 33.3707 | 33.3691 | 33.3386 | 33.3705 | 33.3863 | 33.4005 | 33.3921 | 33.4671 | 33.4303 | 33.4561 | 33.4238 | 33.4454 | 33.4705 | 33.4259 |
|          6 | 33.3892 | 33.3892 | 33.3736 | 33.3703 | 33.3909 | 33.3995 | 33.3611 | 33.4016 | 33.3569 | 33.3952 | 33.4358 | 33.4369 | 33.3807 | 33.3557 | 33.4143 | 33.4316 | 33.3757 | 33.4437 | 33.4106 | 33.3799 | 33.4453 |
|          7 | 33.4079 | 33.4079 | 33.4096 | 33.3708 | 33.3732 | 33.4352 | 33.3685 | 33.4174 | 33.3749 | 33.3592 | 33.3759 | 33.3719 | 33.3719 | 33.406  | 33.4148 | 33.413  | 33.4833 | 33.4295 | 33.4    | 33.432  | 33.4344 |
|          8 | 33.4307 | 33.4307 | 33.424  | 33.4322 | 33.4328 | 33.4255 | 33.4285 | 33.4583 | 33.4552 | 33.4864 | 33.4074 | 33.4436 | 33.4404 | 33.4668 | 33.4403 | 33.4846 | 33.4414 | 33.4862 | 33.4164 | 33.4003 | 33.4516 |
|          9 | 33.4534 | 33.4534 | 33.4659 | 33.4885 | 33.4921 | 33.4931 | 33.447  | 33.4592 | 33.4561 | 33.4665 | 33.4584 | 33.495  | 33.4624 | 33.4863 | 33.4723 | 33.4697 | 33.4422 | 33.408  | 33.4346 | 33.4421 | 33.4491 |
|         10 | 33.5507 | 33.5507 | 33.5214 | 33.5299 | 33.5191 | 33.5004 | 33.487  | 33.4929 | 33.4756 | 33.456  | 33.4863 | 33.4713 | 33.5165 | 33.5255 | 33.4961 | 33.4933 | 33.5118 | 33.4995 | 33.5176 | 33.4718 | 33.4731 |
|         11 | 33.5695 | 33.5695 | 33.5757 | 33.5869 | 33.5676 | 33.5708 | 33.5466 | 33.5801 | 33.5533 | 33.5287 | 33.5348 | 33.5243 | 33.503  | 33.5165 | 33.5244 | 33.5233 | 33.5025 | 33.5063 | 33.515  | 33.4476 | 33.4701 |
|         12 | 33.6293 | 33.6293 | 33.611  | 33.6109 | 33.5818 | 33.5824 | 33.6033 | 33.5717 | 33.5613 | 33.5442 | 33.5601 | 33.5358 | 33.554  | 33.5385 | 33.5246 | 33.5093 | 33.4836 | 33.5031 | 33.5172 | 33.5111 | 33.5045 |
|         13 | 33.6542 | 33.6542 | 33.6747 | 33.6738 | 33.6467 | 33.6241 | 33.6047 | 33.6123 | 33.5673 | 33.5393 | 33.5787 | 33.5522 | 33.554  | 33.5475 | 33.5443 | 33.531  | 33.5103 | 33.566  | 33.5067 | 33.5323 | 33.5162 |
|         14 | 33.7149 | 33.7149 | 33.6866 | 33.6836 | 33.6596 | 33.6767 | 33.6558 | 33.6314 | 33.5863 | 33.5841 | 33.6022 | 33.594  | 33.5899 | 33.5602 | 33.5816 | 33.5689 | 33.5581 | 33.5432 | 33.5567 | 33.5275 | 33.5214 |
|         15 | 33.7393 | 33.7393 | 33.7375 | 33.7327 | 33.6992 | 33.6806 | 33.6934 | 33.6556 | 33.6447 | 33.6143 | 33.6132 | 33.6127 | 33.6086 | 33.5808 | 33.5665 | 33.5687 | 33.5709 | 33.5748 | 33.574  | 33.556  | 33.5271 |
|         16 | 33.768  | 33.768  | 33.7514 | 33.7454 | 33.7406 | 33.7076 | 33.6935 | 33.6754 | 33.6592 | 33.6462 | 33.6532 | 33.6451 | 33.6365 | 33.6146 | 33.614  | 33.6104 | 33.6015 | 33.606  | 33.5725 | 33.5388 | 33.5671 |
|         17 | 33.7885 | 33.7885 | 33.7588 | 33.7745 | 33.7625 | 33.7459 | 33.7051 | 33.6798 | 33.6998 | 33.6614 | 33.633  | 33.657  | 33.6537 | 33.6208 | 33.5931 | 33.607  | 33.5838 | 33.5744 | 33.5782 | 33.5908 | 33.5569 |
|         18 | 33.807  | 33.807  | 33.7894 | 33.7871 | 33.778  | 33.744  | 33.7308 | 33.7317 | 33.6929 | 33.6514 | 33.6489 | 33.6754 | 33.659  | 33.6369 | 33.6282 | 33.6157 | 33.6272 | 33.6008 | 33.5789 | 33.5786 | 33.5198 |
|         19 | 33.8212 | 33.8212 | 33.7946 | 33.794  | 33.7777 | 33.773  | 33.7593 | 33.735  | 33.7148 | 33.6743 | 33.6882 | 33.6866 | 33.6644 | 33.6597 | 33.6477 | 33.6152 | 33.615  | 33.6323 | 33.6061 | 33.5819 | 33.568  |
|         20 | 33.8215 | 33.8215 | 33.8038 | 33.7979 | 33.7863 | 33.7798 | 33.7667 | 33.7582 | 33.7367 | 33.6962 | 33.6927 | 33.6809 | 33.6708 | 33.6528 | 33.6529 | 33.6393 | 33.6279 | 33.6092 | 33.6012 | 33.6135 | 33.5706 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 34.3006 | 34.3006 | 34.3006 | 34.3006 | 34.3006 | 34.3006 | 34.3006 | 34.3006 | 34.3006 | 34.172  |
|          0.2 | 34.1455 | 34.1455 | 34.1455 | 34.1455 | 34.1455 | 34.1455 | 34.1455 | 34.1455 | 34.1455 | 33.9    |
|          0.3 | 33.8342 | 33.8342 | 33.8342 | 33.8342 | 33.8342 | 33.8342 | 33.8342 | 33.8342 | 33.8342 | 33.8363 |
|          0.4 | 33.8271 | 33.8271 | 33.8271 | 33.8271 | 33.8271 | 33.8271 | 33.8271 | 33.8271 | 33.8271 | 33.611  |
|          0.5 | 33.8046 | 33.8046 | 33.8046 | 33.8046 | 33.8046 | 33.8046 | 33.8046 | 33.8046 | 33.8046 | 33.5967 |
|          0.6 | 33.6294 | 33.6294 | 33.6294 | 33.6294 | 33.6294 | 33.6294 | 33.6294 | 33.6294 | 33.6294 | 33.5498 |
|          0.7 | 33.6193 | 33.6193 | 33.6193 | 33.6193 | 33.6193 | 33.6193 | 33.6193 | 33.6193 | 33.6193 | 33.426  |
|          0.8 | 33.5805 | 33.5805 | 33.5805 | 33.5805 | 33.5805 | 33.5805 | 33.5805 | 33.5805 | 33.5805 | 33.4002 |
|          0.9 | 33.4719 | 33.4719 | 33.4719 | 33.4719 | 33.4719 | 33.4719 | 33.4719 | 33.4719 | 33.4719 | 33.3897 |
|          1   | 33.4979 | 33.4979 | 33.4979 | 33.4979 | 33.4979 | 33.4979 | 33.498  | 33.4979 | 33.4979 | 33.3386 |

***

### eta Grid Search

| eta   |     0.5 |    0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|---------|--------|---------|---------|---------|---------|---------|
| MAE   | 33.4381 | 33.371 | 33.3386 | 33.4265 | 33.3866 | 33.3936 | 62.6692 |

***

### Tuned Parameters

- max_depth = 5
- min_child_weight = 9
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 45