# T2 Week 2 Search (Deterministic + Strict)

- Date: 2026-02-07T19:44:54.365298+00:00
- Search space: expanded
- Budget: 6 configs x [1200, 1280, 1400, 1536, 1600, 1792, 1920, 2048] sizes x 12 runs
- Input distribution: standard_normal
- Correctness threshold: 0.001

## Search Results

| Rank | Config | Family | Vec | Unroll | Local | Size | GFLOPS | Max Error | Delta vs Baseline | Status |
| ---: | --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| 1 | t24_prod_v4_u0_l12 | tile24 | 4 | 0 | 12x12 | 1536 | 795.765 | 0.000366 | +107.800% | valid |
| 2 | t20_v3vec_v4_u0_l10 | tile20 | 4 | 0 | 10x10 | 1536 | 615.643 | 0.000412 | +60.764% | valid |
| 3 | t20_regblock_v4_u0_l5 | tile20 | 4 | 0 | 5x5 | 1536 | 566.595 | 0.000320 | +47.957% | valid |
| 4 | t20_v3vec_v4_u0_l10 | tile20 | 4 | 0 | 10x10 | 1792 | 801.747 | 0.000443 | +40.352% | valid |
| 5 | t24_prod_v4_u0_l12 | tile24 | 4 | 0 | 12x12 | 1792 | 788.669 | 0.000473 | +38.063% | valid |
| 6 | t20_prefetch_v4_u4_l10 | tile20 | 4 | 4 | 10x10 | 1536 | 506.727 | 0.000351 | +32.323% | valid |
| 7 | t20_prefetch_v4_u4_l10 | tile20 | 4 | 4 | 10x10 | 1792 | 754.631 | 0.000443 | +32.104% | valid |
| 8 | t20_v3vec_v4_u0_l10 | tile20 | 4 | 0 | 10x10 | 1280 | 835.914 | 0.000282 | +32.061% | valid |
| 9 | t20_v3vec_v4_u0_l10 | tile20 | 4 | 0 | 10x10 | 1600 | 946.075 | 0.000351 | +26.887% | valid |
| 10 | t20_prefetch_v4_u4_l10 | tile20 | 4 | 4 | 10x10 | 1280 | 792.199 | 0.000290 | +25.155% | valid |
| 11 | t20_v3vec_v4_u0_l10 | tile20 | 4 | 0 | 10x10 | 1200 | 916.614 | 0.000229 | +23.846% | valid |
| 12 | t24_prod_v4_u0_l12 | tile24 | 4 | 0 | 12x12 | 1280 | 763.435 | 0.000259 | +20.610% | valid |
| 13 | t20_v3vec_v4_u0_l10 | tile20 | 4 | 0 | 10x10 | 1400 | 931.264 | 0.000397 | +16.385% | valid |
| 14 | t20_v3vec_v4_u0_l10 | tile20 | 4 | 0 | 10x10 | 1920 | 920.433 | 0.000519 | +14.863% | valid |
| 15 | t20_prefetch_v4_u4_l10 | tile20 | 4 | 4 | 10x10 | 1600 | 845.837 | 0.000328 | +13.443% | valid |
| 16 | t20_prefetch_v4_u4_l10 | tile20 | 4 | 4 | 10x10 | 1200 | 826.720 | 0.000275 | +11.700% | valid |
| 17 | t20_regblock_v4_u0_l5 | tile20 | 4 | 0 | 5x5 | 1792 | 608.893 | 0.000458 | +6.592% | valid |
| 18 | t24_prod_v4_u0_l12 | tile24 | 4 | 0 | 12x12 | 1200 | 785.183 | 0.000259 | +6.088% | valid |
| 19 | t24_prod_v4_u0_l12 | tile24 | 4 | 0 | 12x12 | 1600 | 787.200 | 0.000397 | +5.579% | valid |
| 20 | t20_prefetch_v4_u4_l10 | tile20 | 4 | 4 | 10x10 | 1400 | 839.674 | 0.000336 | +4.938% | valid |
| 21 | t20_prefetch_v4_u4_l10 | tile20 | 4 | 4 | 10x10 | 1920 | 840.512 | 0.000435 | +4.889% | valid |
| 22 | t24_prod_v4_u0_l12 | tile24 | 4 | 0 | 12x12 | 1920 | 801.331 | 0.000656 | +0.000% | valid |
| 23 | t20_prod_v4_u10_l10 | tile20 | 4 | 10 | 10x10 | 1400 | 800.159 | 0.000412 | +0.000% | valid |
| 24 | t20_prod_v4_u10_l10 | tile20 | 4 | 10 | 10x10 | 1600 | 745.606 | 0.000336 | +0.000% | valid |
| 25 | t20_prod_v4_u10_l10 | tile20 | 4 | 10 | 10x10 | 1200 | 740.125 | 0.000244 | +0.000% | valid |
| 26 | t24_prod_v4_u0_l12 | tile24 | 4 | 0 | 12x12 | 2048 | 738.786 | 0.000496 | +0.000% | valid |
| 27 | t20_prod_v4_u10_l10 | tile20 | 4 | 10 | 10x10 | 1280 | 632.976 | 0.000282 | +0.000% | valid |
| 28 | t20_prod_v4_u10_l10 | tile20 | 4 | 10 | 10x10 | 1792 | 571.238 | 0.000397 | +0.000% | valid |
| 29 | t20_prod_v4_u10_l10 | tile20 | 4 | 10 | 10x10 | 1536 | 382.947 | 0.000336 | +0.000% | valid |
| 30 | t20_regblock_v4_u0_l5 | tile20 | 4 | 0 | 5x5 | 1280 | 620.683 | 0.000290 | -1.942% | valid |
| 31 | t24_prod_v4_u0_l12 | tile24 | 4 | 0 | 12x12 | 1400 | 769.628 | 0.000305 | -3.816% | valid |
| 32 | t20_prod_v4_u10_l10 | tile20 | 4 | 10 | 10x10 | 1920 | 703.238 | 0.000473 | -12.241% | valid |
| 33 | t20_regblock_v4_u0_l5 | tile20 | 4 | 0 | 5x5 | 1600 | 643.515 | 0.000336 | -13.692% | valid |
| 34 | t20_regblock_v4_u0_l5 | tile20 | 4 | 0 | 5x5 | 1200 | 631.387 | 0.000305 | -14.692% | valid |
| 35 | t20_regblock_v4_u0_l5 | tile20 | 4 | 0 | 5x5 | 1920 | 644.156 | 0.000473 | -19.614% | valid |
| 36 | t20_regblock_v4_u0_l5 | tile20 | 4 | 0 | 5x5 | 1400 | 632.060 | 0.000320 | -21.008% | valid |
| 37 | t20_float8_v8_u8_l10 | tile20 | 8 | 8 | 10x10 | 1536 | 270.746 | 0.000351 | -29.299% | valid |
| 38 | t20_regblock_v4_u0_l5 | tile20 | 4 | 0 | 5x5 | 2048 | 434.643 | 0.000488 | -41.168% | valid |
| 39 | t20_float8_v8_u8_l10 | tile20 | 8 | 8 | 10x10 | 1280 | 304.914 | 0.000267 | -51.828% | valid |
| 40 | t20_v3vec_v4_u0_l10 | tile20 | 4 | 0 | 10x10 | 2048 | 340.189 | 0.000519 | -53.953% | valid |
| 41 | t20_float8_v8_u8_l10 | tile20 | 8 | 8 | 10x10 | 1600 | 310.200 | 0.000381 | -58.396% | valid |
| 42 | t20_float8_v8_u8_l10 | tile20 | 8 | 8 | 10x10 | 1200 | 306.120 | 0.000267 | -58.639% | valid |
| 43 | t20_prod_v4_u10_l10 | tile20 | 4 | 10 | 10x10 | 2048 | 291.014 | 0.000473 | -60.609% | valid |
| 44 | t20_float8_v8_u8_l10 | tile20 | 8 | 8 | 10x10 | 1920 | 311.528 | 0.000412 | -61.124% | valid |
| 45 | t20_float8_v8_u8_l10 | tile20 | 8 | 8 | 10x10 | 1400 | 309.503 | 0.000336 | -61.320% | valid |
| 46 | t20_prefetch_v4_u4_l10 | tile20 | 4 | 4 | 10x10 | 2048 | 231.598 | 0.000534 | -68.652% | valid |

## Replay Summary

| Candidate | Vec | Unroll | Local | Mean GFLOPS | CV | Max Error (max) | Delta vs Baseline | Correctness | Stability | Promotion |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| t24_prod_v4_u0_l12@1536 | 4 | 0 | 12x12 | 795.828 | 0.00027 | 0.000397 | +107.817% | True | True | True |
| t20_v3vec_v4_u0_l10@1536 | 4 | 0 | 10x10 | 614.311 | 0.00231 | 0.000412 | +60.417% | True | True | True |
| t20_regblock_v4_u0_l5@1536 | 4 | 0 | 5x5 | 565.396 | 0.00194 | 0.000443 | +47.643% | True | True | True |
| t20_v3vec_v4_u0_l10@1792 | 4 | 0 | 10x10 | 797.051 | 0.00355 | 0.000504 | +39.530% | True | True | True |

## Decision Hint

- Suggested decision: `promote`
- Decision rationale: At least one replayed candidate passed performance, correctness and stability gates.
