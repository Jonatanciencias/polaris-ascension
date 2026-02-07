# T3 Week 3 Block 3 - Strict Shadow Policy Report

- Date: 2026-02-07T20:18:56.879743+00:00
- Protocol: epochs=3, runs_per_decision=8, warmup=2, seed=42
- Policy: epsilon=0.05, bootstrap_weight=3.0, warmup_steps_per_context=2
- Guardrails: regression_limit=0.08, max_fallback_rate=0.1, min_delta_for_nonstatic=8.0, freeze_sizes=[1400, 2048]
- Bootstrap source: research/breakthrough_lab/t2_auto_scheduler/week2_t2_expanded_search_20260207_194454.json

## Summary

- Static mean GFLOPS: 783.033
- Shadow mean GFLOPS: 866.277
- Mean uplift vs static: +10.631%
- P95 latency delta vs static: +0.000%
- Fallback rate: 0.000
- Exploration rate: 0.000
- Correctness failures: 0
- Stop rule triggered: False
- Decision hint: promote

## Per-Size Delta

| Size | Static Mean | Shadow Mean | Delta | Samples |
| ---: | ---: | ---: | ---: | ---: |
| 1200 | 738.843 | 915.519 | +23.913% | 3 |
| 1280 | 635.574 | 836.711 | +31.647% | 3 |
| 1400 | 929.432 | 929.432 | +0.000% | 3 |
| 1536 | 795.438 | 795.438 | +0.000% | 3 |
| 1600 | 787.503 | 946.290 | +20.163% | 3 |
| 1792 | 788.670 | 799.628 | +1.389% | 3 |
| 1920 | 802.670 | 921.068 | +14.750% | 3 |
| 2048 | 786.134 | 786.134 | +0.000% | 3 |

## Fallback Reasons

```json
{}
```

## Policy Snapshot

```json
{
  "size_1200": {
    "steps": 3,
    "arms": {
      "tile20": {
        "count": 3,
        "mean_reward": 740.1252387846184
      },
      "tile20_v3_1400": {
        "count": 6,
        "mean_reward": 916.0663628674143
      },
      "tile24": {
        "count": 0,
        "mean_reward": 0.0
      }
    }
  },
  "size_1280": {
    "steps": 3,
    "arms": {
      "tile20": {
        "count": 3,
        "mean_reward": 632.9763730094533
      },
      "tile20_v3_1400": {
        "count": 6,
        "mean_reward": 836.312379278203
      },
      "tile24": {
        "count": 3,
        "mean_reward": 763.4354166616107
      }
    }
  },
  "size_1400": {
    "steps": 3,
    "arms": {
      "tile20": {
        "count": 0,
        "mean_reward": 0.0
      },
      "tile20_v3_1400": {
        "count": 6,
        "mean_reward": 930.3480697904982
      },
      "tile24": {
        "count": 0,
        "mean_reward": 0.0
      }
    }
  },
  "size_1536": {
    "steps": 3,
    "arms": {
      "tile20": {
        "count": 0,
        "mean_reward": 0.0
      },
      "tile20_v3_1400": {
        "count": 3,
        "mean_reward": 615.6427865677558
      },
      "tile24": {
        "count": 6,
        "mean_reward": 795.6016155853508
      }
    }
  },
  "size_1600": {
    "steps": 3,
    "arms": {
      "tile20": {
        "count": 0,
        "mean_reward": 0.0
      },
      "tile20_v3_1400": {
        "count": 6,
        "mean_reward": 946.1826445070801
      },
      "tile24": {
        "count": 3,
        "mean_reward": 787.1998032000491
      }
    }
  },
  "size_1792": {
    "steps": 3,
    "arms": {
      "tile20": {
        "count": 0,
        "mean_reward": 0.0
      },
      "tile20_v3_1400": {
        "count": 6,
        "mean_reward": 800.6873424589181
      },
      "tile24": {
        "count": 3,
        "mean_reward": 788.6688130603653
      }
    }
  },
  "size_1920": {
    "steps": 3,
    "arms": {
      "tile20": {
        "count": 0,
        "mean_reward": 0.0
      },
      "tile20_v3_1400": {
        "count": 6,
        "mean_reward": 920.7507108262681
      },
      "tile24": {
        "count": 3,
        "mean_reward": 801.3314222442617
      }
    }
  },
  "size_2048": {
    "steps": 3,
    "arms": {
      "tile20": {
        "count": 0,
        "mean_reward": 0.0
      },
      "tile20_v3_1400": {
        "count": 0,
        "mean_reward": 0.0
      },
      "tile24": {
        "count": 6,
        "mean_reward": 762.4598136112021
      }
    }
  }
}
```
