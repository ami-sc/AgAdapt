# MNH1 - Grain_Yield Model Summary [v0.1__XGB_LD_ATF]

***

### Model Performance

- Baseline Model [MAE] = 27.8416
- Baseline Model [RMSE] = 38.0609
- Trained Model [MAE] = 35.7905
- Trained Model [RMSE] = 44.7991
- Prediction [MAE] = 25.9978
- Prediction [RMSE] = 37.1762
***

### Dataset Statistics

- LOFO Field [Mean] = 146.2302 [bu/A]
- LOFO Field [Standard Deviation] = 37.0035 [bu/A]
- Model Dataset [Mean] = 155.3298 [bu/A]
- Model Dataset [Standard Deviation] = 45.8728 [bu/A]
***

### max_depth & min_child_weight Grid Search

|   md \ mcw |       0 |       1 |       2 |       3 |       4 |       5 |       6 |       7 |       8 |       9 |      10 |      11 |      12 |      13 |      14 |      15 |      16 |      17 |      18 |      19 |      20 |
|------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          1 | 35.8419 | 35.8419 | 35.8419 | 35.8419 | 35.8419 | 35.8419 | 35.8419 | 35.8407 | 35.7982 | 35.8185 | 35.7989 | 35.7966 | 35.8048 | 35.8302 | 35.8302 | 35.8302 | 35.8286 | 35.8127 | 35.8127 | 35.8127 | 35.8127 |
|          2 | 35.4535 | 35.4535 | 35.4585 | 35.4585 | 35.4585 | 35.4767 | 35.4055 | 35.313  | 35.2839 | 35.4485 | 35.4858 | 35.3644 | 35.3365 | 35.3156 | 35.4895 | 35.4645 | 35.3561 | 35.3819 | 35.4858 | 35.4203 | 35.429  |
|          3 | 35.2103 | 35.2103 | 35.21   | 35.1592 | 35.2147 | 35.1652 | 35.1923 | 35.243  | 35.197  | 35.238  | 35.2142 | 35.2061 | 35.1856 | 35.4308 | 35.2703 | 35.2306 | 35.2324 | 35.257  | 35.314  | 35.3858 | 35.2589 |
|          4 | 35.1716 | 35.1716 | 35.1586 | 35.1503 | 35.1883 | 35.1394 | 35.169  | 35.1813 | 35.1508 | 35.1845 | 35.1885 | 35.1919 | 35.242  | 35.1908 | 35.1652 | 35.1579 | 35.2172 | 35.1942 | 35.1844 | 35.2139 | 35.2013 |
|          5 | 35.1608 | 35.1608 | 35.1641 | 35.1374 | 35.137  | 35.1448 | 35.1232 | 35.1315 | 35.1122 | 35.1444 | 35.1539 | 35.1705 | 35.1809 | 35.2023 | 35.2135 | 35.1853 | 35.1561 | 35.2049 | 35.1926 | 35.2027 | 35.1619 |
|          6 | 35.1988 | 35.1988 | 35.1624 | 35.1514 | 35.1458 | 35.1655 | 35.146  | 35.1326 | 35.13   | 35.1428 | 35.1189 | 35.1388 | 35.162  | 35.1404 | 35.151  | 35.164  | 35.1671 | 35.1686 | 35.1667 | 35.1703 | 35.1617 |
|          7 | 35.18   | 35.18   | 35.1915 | 35.1759 | 35.1536 | 35.1975 | 35.1669 | 35.1708 | 35.1884 | 35.1773 | 35.1758 | 35.1647 | 35.1927 | 35.1529 | 35.1428 | 35.1754 | 35.1606 | 35.1778 | 35.1397 | 35.1874 | 35.1559 |
|          8 | 35.1883 | 35.1883 | 35.163  | 35.2058 | 35.189  | 35.1569 | 35.1807 | 35.1626 | 35.1837 | 35.1473 | 35.1795 | 35.1385 | 35.1395 | 35.1979 | 35.1411 | 35.1657 | 35.1258 | 35.1616 | 35.1531 | 35.2    | 35.1726 |
|          9 | 35.2486 | 35.2486 | 35.2232 | 35.2341 | 35.1955 | 35.21   | 35.2128 | 35.1866 | 35.1784 | 35.1812 | 35.2212 | 35.2043 | 35.1935 | 35.1849 | 35.1663 | 35.1966 | 35.2056 | 35.1996 | 35.1608 | 35.16   | 35.1631 |
|         10 | 35.2782 | 35.2782 | 35.2344 | 35.2544 | 35.2638 | 35.2411 | 35.207  | 35.1781 | 35.2011 | 35.2263 | 35.1788 | 35.1801 | 35.1795 | 35.2078 | 35.1688 | 35.1564 | 35.1723 | 35.1808 | 35.1735 | 35.167  | 35.1539 |
|         11 | 35.2807 | 35.2807 | 35.2839 | 35.2374 | 35.2699 | 35.2469 | 35.2542 | 35.2522 | 35.2298 | 35.2173 | 35.2357 | 35.2012 | 35.2372 | 35.1901 | 35.2036 | 35.2356 | 35.1883 | 35.1743 | 35.2147 | 35.147  | 35.1507 |
|         12 | 35.3453 | 35.3453 | 35.3194 | 35.3111 | 35.2982 | 35.2754 | 35.2469 | 35.2556 | 35.2496 | 35.2321 | 35.2402 | 35.2308 | 35.2568 | 35.2441 | 35.2266 | 35.2301 | 35.211  | 35.1952 | 35.1921 | 35.1608 | 35.1562 |
|         13 | 35.3676 | 35.3676 | 35.3326 | 35.3389 | 35.3224 | 35.3126 | 35.2931 | 35.2875 | 35.2984 | 35.269  | 35.2531 | 35.2323 | 35.2453 | 35.238  | 35.2514 | 35.2352 | 35.2217 | 35.1987 | 35.1892 | 35.1751 | 35.1946 |
|         14 | 35.4129 | 35.4129 | 35.3527 | 35.3758 | 35.3463 | 35.3428 | 35.342  | 35.2938 | 35.3032 | 35.3021 | 35.2929 | 35.2604 | 35.2635 | 35.2356 | 35.2557 | 35.2683 | 35.2662 | 35.2151 | 35.2115 | 35.2003 | 35.1912 |
|         15 | 35.4123 | 35.4123 | 35.402  | 35.3757 | 35.3547 | 35.3541 | 35.3415 | 35.3314 | 35.2766 | 35.3188 | 35.3013 | 35.2811 | 35.2912 | 35.247  | 35.2501 | 35.2435 | 35.2528 | 35.2314 | 35.2286 | 35.2054 | 35.2054 |
|         16 | 35.4482 | 35.4482 | 35.4199 | 35.4042 | 35.3945 | 35.3713 | 35.3642 | 35.3327 | 35.3208 | 35.2939 | 35.3193 | 35.2968 | 35.2725 | 35.2725 | 35.2958 | 35.2769 | 35.2857 | 35.2521 | 35.2447 | 35.2073 | 35.2097 |
|         17 | 35.4566 | 35.4566 | 35.4183 | 35.4205 | 35.4083 | 35.3928 | 35.3743 | 35.3338 | 35.3297 | 35.316  | 35.2991 | 35.2996 | 35.3052 | 35.305  | 35.289  | 35.2602 | 35.2753 | 35.2552 | 35.2524 | 35.2116 | 35.2343 |
|         18 | 35.4714 | 35.4714 | 35.4707 | 35.4441 | 35.4354 | 35.4053 | 35.3854 | 35.3589 | 35.3716 | 35.3427 | 35.3245 | 35.3338 | 35.3155 | 35.2871 | 35.284  | 35.2678 | 35.2832 | 35.2587 | 35.2443 | 35.2256 | 35.232  |
|         19 | 35.5053 | 35.5053 | 35.446  | 35.4391 | 35.4283 | 35.4153 | 35.412  | 35.3784 | 35.3744 | 35.3394 | 35.3457 | 35.3512 | 35.3254 | 35.297  | 35.3178 | 35.3102 | 35.2679 | 35.2783 | 35.2409 | 35.2535 | 35.2383 |
|         20 | 35.5016 | 35.5016 | 35.4719 | 35.4705 | 35.4743 | 35.4377 | 35.4228 | 35.3931 | 35.4022 | 35.3659 | 35.3483 | 35.35   | 35.3318 | 35.3157 | 35.3269 | 35.2908 | 35.3179 | 35.2572 | 35.2412 | 35.2601 | 35.2364 |

***

### subsample & colsample_bytree Grid Search

|   ssmpl \ cb |     0.1 |     0.2 |     0.3 |     0.4 |     0.5 |     0.6 |     0.7 |     0.8 |     0.9 |     1.0 |
|--------------|---------|---------|---------|---------|---------|---------|---------|---------|---------|---------|
|          0.1 | 36.1322 | 36.1322 | 36.1322 | 36.1322 | 36.1322 | 36.1322 | 36.1322 | 36.1322 | 36.1322 | 35.9878 |
|          0.2 | 35.7933 | 35.7933 | 35.7933 | 35.7933 | 35.7933 | 35.7933 | 35.7933 | 35.7933 | 35.7933 | 35.5933 |
|          0.3 | 35.6521 | 35.6521 | 35.6521 | 35.6521 | 35.6521 | 35.6521 | 35.6521 | 35.6521 | 35.6521 | 35.4204 |
|          0.4 | 35.4149 | 35.4149 | 35.4149 | 35.4149 | 35.4149 | 35.4149 | 35.4149 | 35.4149 | 35.4149 | 35.4083 |
|          0.5 | 35.4062 | 35.4062 | 35.4062 | 35.4062 | 35.4062 | 35.4062 | 35.4062 | 35.4062 | 35.4062 | 35.1903 |
|          0.6 | 35.3558 | 35.3558 | 35.3558 | 35.3558 | 35.3558 | 35.3558 | 35.3558 | 35.3558 | 35.3558 | 35.2756 |
|          0.7 | 35.2677 | 35.2677 | 35.2677 | 35.2677 | 35.2677 | 35.2677 | 35.2677 | 35.2677 | 35.2677 | 35.2258 |
|          0.8 | 35.2769 | 35.2769 | 35.2769 | 35.2769 | 35.2769 | 35.2769 | 35.2769 | 35.2769 | 35.2769 | 35.1787 |
|          0.9 | 35.282  | 35.282  | 35.2819 | 35.282  | 35.282  | 35.282  | 35.282  | 35.282  | 35.282  | 35.2048 |
|          1   | 35.2578 | 35.2578 | 35.2578 | 35.2578 | 35.2578 | 35.2578 | 35.2578 | 35.2578 | 35.2578 | 35.1122 |

***

### eta Grid Search

| eta   |     0.5 |     0.4 |     0.3 |    0.2 |     0.1 |   0.01 |   0.001 |
|-------|---------|---------|---------|--------|---------|--------|---------|
| MAE   | 35.1158 | 35.1612 | 35.1122 | 35.139 | 35.2052 | 35.205 | 62.9435 |

***

### Tuned Parameters

- max_depth = 5
- min_child_weight = 8
- subsample = 1.0
- colsample_bytree = 1.0
- eta = 0.3
- num_boost_round = 73