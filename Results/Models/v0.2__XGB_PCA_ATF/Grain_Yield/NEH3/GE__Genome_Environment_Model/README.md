# NEH3 - Grain_Yield Model Summary [v0.2__XGB_PCA_ATF]

***

### Model Performance

- Baseline Model [MAE] = 41.2876
- Baseline Model [RMSE] = 53.4464
- Trained Model [MAE] = 21.6793
- Trained Model [RMSE] = 28.5751
- Prediction [MAE] = 32.5949
- Prediction [RMSE] = 42.1333
***

### Dataset Statistics

- LOFO Field [Mean] = 118.8600 [bu/A]
- LOFO Field [Standard Deviation] = 38.9670 [bu/A]
- Model Dataset [Mean] = 155.5743 [bu/A]
- Model Dataset [Standard Deviation] = 45.3664 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 24.2506 | 24.2506 | 24.2506 | 24.2506 | 24.2506 | 24.2553 | 24.2565 | 24.3694 | 24.2472 | 24.2472 | 24.2472 | 24.2494 | 24.2499 | 24.2499 | 24.2737 | 24.2737 | 24.2737 | 24.2737 | 24.2737 | 24.2832 | 24.2832 |
|          2 | 22.6542 | 22.6542 | 22.8088 | 22.6219 | 22.5695 | 22.5466 | 22.8397 | 22.6951 | 22.6703 | 22.7948 | 22.7226 | 22.6668 | 22.8935 | 23.0586 | 22.759  | 22.7902 | 22.7804 | 22.9219 | 22.7526 | 22.7931 | 22.9481 |
|          3 | 22.5215 | 22.5215 | 22.4942 | 22.5287 | 22.5007 | 22.477  | 22.4738 | 22.5058 | 22.4207 | 22.2897 | 22.5166 | 22.5329 | 22.5059 | 22.4773 | 22.5891 | 22.5847 | 22.5988 | 22.5534 | 22.5488 | 22.6041 | 22.5806 |
|          4 | 22.517  | 22.517  | 22.4701 | 22.4716 | 22.4121 | 22.4229 | 22.48   | 22.3987 | 22.5034 | 22.4031 | 22.5041 | 22.5043 | 22.5252 | 22.4663 | 22.4322 | 22.6273 | 22.4349 | 22.4229 | 22.4973 | 22.503  | 22.524  |
|          5 | 22.4392 | 22.4392 | 22.4548 | 22.4276 | 22.3857 | 22.5555 | 22.4886 | 22.4598 | 22.4792 | 22.4327 | 22.463  | 22.4095 | 22.4244 | 22.4028 | 22.4105 | 22.6116 | 22.5804 | 22.5182 | 22.5098 | 22.5017 | 22.4817 |
|          6 | 22.517  | 22.517  | 22.518  | 22.4456 | 22.3235 | 22.4355 | 22.4634 | 22.3984 | 22.4217 | 22.5463 | 22.3884 | 22.4968 | 22.4815 | 22.414  | 22.4595 | 22.5937 | 22.4921 | 22.4954 | 22.4403 | 22.5521 | 22.4663 |
|          7 | 22.5298 | 22.5298 | 22.484  | 22.521  | 22.3855 | 22.5353 | 22.3703 | 22.3945 | 22.3812 | 22.4336 | 22.4007 | 22.3782 | 22.4348 | 22.4953 | 22.491  | 22.5694 | 22.5065 | 22.4308 | 22.4337 | 22.4166 | 22.4831 |
|          8 | 22.6029 | 22.6029 | 22.6126 | 22.4914 | 22.3961 | 22.4247 | 22.5623 | 22.4643 | 22.5819 | 22.5321 | 22.4651 | 22.4585 | 22.4536 | 22.5248 | 22.6075 | 22.4574 | 22.6233 | 22.4983 | 22.5928 | 22.46   | 22.4909 |
|          9 | 22.7166 | 22.7166 | 22.7069 | 22.7713 | 22.6385 | 22.4665 | 22.5635 | 22.5243 | 22.5421 | 22.6568 | 22.4359 | 22.5243 | 22.6111 | 22.603  | 22.5539 | 22.5216 | 22.5689 | 22.5376 | 22.5205 | 22.4839 | 22.5399 |
|         10 | 22.8806 | 22.8806 | 22.7863 | 22.7937 | 22.6723 | 22.7183 | 22.4877 | 22.5867 | 22.6455 | 22.5927 | 22.6414 | 22.5167 | 22.5428 | 22.5396 | 22.5241 | 22.4546 | 22.5534 | 22.5083 | 22.5382 | 22.4    | 22.5229 |
|         11 | 22.9919 | 22.9919 | 23.0036 | 22.7581 | 22.8453 | 22.7657 | 22.796  | 22.6212 | 22.7242 | 22.5637 | 22.6922 | 22.5375 | 22.6313 | 22.5751 | 22.5746 | 22.5845 | 22.5832 | 22.6462 | 22.5966 | 22.6477 | 22.6353 |
|         12 | 23.1462 | 23.1462 | 23.1087 | 23.0096 | 22.8991 | 22.8089 | 22.7366 | 22.7295 | 22.7034 | 22.6622 | 22.6864 | 22.6465 | 22.7451 | 22.7255 | 22.7163 | 22.5867 | 22.6005 | 22.5341 | 22.5818 | 22.6357 | 22.6905 |
|         13 | 23.4349 | 23.4349 | 23.2378 | 23.0461 | 22.9279 | 22.7631 | 22.7953 | 22.7881 | 22.8416 | 22.6881 | 22.6228 | 22.6899 | 22.7309 | 22.6322 | 22.6718 | 22.6701 | 22.6396 | 22.6391 | 22.7287 | 22.6227 | 22.5705 |
|         14 | 23.6192 | 23.6192 | 23.4489 | 23.2067 | 22.9744 | 22.9015 | 22.9479 | 22.7912 | 22.8272 | 22.8189 | 22.7255 | 22.7548 | 22.7565 | 22.6699 | 22.6732 | 22.6916 | 22.6199 | 22.7093 | 22.6225 | 22.5985 | 22.6594 |
|         15 | 23.7926 | 23.7926 | 23.5726 | 23.3759 | 23.1544 | 22.9763 | 23.0672 | 22.784  | 22.8955 | 22.869  | 22.6962 | 22.6686 | 22.7429 | 22.7309 | 22.6718 | 22.6197 | 22.7828 | 22.6936 | 22.7698 | 22.6467 | 22.6975 |
|         16 | 23.8603 | 23.8603 | 23.7005 | 23.3958 | 23.169  | 22.9921 | 23.1304 | 22.8269 | 22.8915 | 22.8653 | 22.8238 | 22.7605 | 22.781  | 22.8041 | 22.681  | 22.6478 | 22.7475 | 22.7347 | 22.6589 | 22.7135 | 22.6642 |
|         17 | 23.9948 | 23.9948 | 23.7687 | 23.4365 | 23.2529 | 23.2046 | 23.1565 | 22.8708 | 22.9007 | 22.9089 | 22.7404 | 22.8689 | 22.7097 | 22.7548 | 22.7076 | 22.7801 | 22.6614 | 22.6159 | 22.7263 | 22.7484 | 22.678  |
|         18 | 24.0835 | 24.0835 | 23.8466 | 23.5029 | 23.3    | 23.1993 | 23.2403 | 22.8731 | 23.0192 | 22.913  | 22.8805 | 22.7901 | 22.6532 | 22.7364 | 22.7802 | 22.7095 | 22.694  | 22.7263 | 22.6283 | 22.6005 | 22.6821 |
|         19 | 24.2019 | 24.2019 | 23.8929 | 23.546  | 23.3536 | 23.2858 | 23.2175 | 22.9485 | 22.8937 | 22.866  | 22.9455 | 22.7672 | 22.8692 | 22.7446 | 22.6977 | 22.7671 | 22.6731 | 22.6828 | 22.5848 | 22.6341 | 22.6438 |
|         20 | 24.2747 | 24.2747 | 23.9245 | 23.6589 | 23.4744 | 23.3212 | 23.3375 | 22.9234 | 22.8954 | 22.8615 | 22.8447 | 22.7834 | 22.8485 | 22.7489 | 22.6746 | 22.7038 | 22.6929 | 22.6784 | 22.6595 | 22.6999 | 22.5976 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 24.6956 | 24.9099 | 24.8653 | 24.8331 | 24.8893 | 24.9056 | 24.8062 | 25.0138 | 24.9157 | 24.7995 |
|          0.2 | 24.5922 | 24.3168 | 24.2515 | 24.2944 | 24.2552 | 24.3727 | 24.1501 | 24.1682 | 24.2011 | 24.0595 |
|          0.3 | 24.1177 | 24.0647 | 24.1166 | 23.8906 | 24.0132 | 23.9234 | 23.9125 | 24.0027 | 23.9352 | 23.609  |
|          0.4 | 23.8322 | 23.6243 | 23.9367 | 23.5704 | 23.5556 | 23.3831 | 23.4409 | 23.6557 | 23.6097 | 23.4234 |
|          0.5 | 23.8787 | 23.7448 | 23.4238 | 23.267  | 23.5802 | 23.3414 | 23.6343 | 23.4222 | 23.3067 | 23.383  |
|          0.6 | 23.6503 | 23.2639 | 23.4513 | 23.0321 | 23.1399 | 23.0886 | 23.0878 | 22.8517 | 23.1448 | 23.1309 |
|          0.7 | 23.6172 | 22.9986 | 23.1572 | 22.8659 | 23.0401 | 23.0257 | 22.8766 | 22.94   | 22.9675 | 22.8866 |
|          0.8 | 23.1565 | 23.3082 | 22.8275 | 22.774  | 22.6677 | 22.7514 | 22.7699 | 22.8523 | 22.7875 | 22.8767 |
|          0.9 | 23.2254 | 22.7296 | 22.9588 | 22.548  | 22.8534 | 22.7095 | 22.7418 | 22.6142 | 22.5535 | 22.4684 |
|          1   | 23.0203 | 22.8594 | 22.5981 | 22.5293 | 22.4781 | 22.4445 | 22.7407 | 22.3988 | 22.5789 | 22.2897 |

***

### eta Grid Search

| eta   |    0.5 |     0.4 |     0.3 |     0.2 |     0.1 |    0.01 |   0.001 |
|-------|--------|---------|---------|---------|---------|---------|---------|
| MAE   | 22.695 | 22.5909 | 22.2897 | 22.4854 | 22.7273 | 23.7937 | 59.3023 |

***

### Tuned Parameters

- max_depth = 3
- min_child_weight = 9
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 325