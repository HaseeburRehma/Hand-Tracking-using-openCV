# Rules versus trained models: results

- Frames recorded: 7500 (thumb 1500, index 1500, middle 1500, pinky 1500, none 1500)
- People: haseeb · sessions: 3 · conditions: far, normal, normal2
- Split: leave-one-session-out over 3 sessions; 7500 test frames

| Method | Accuracy | Macro F1 | recall thumb | recall index | recall middle | recall pinky | recall none | ms per frame |
|---|---|---|---|---|---|---|---|---|
| rules | 75.8% | 0.687 | 98.7% | 99.8% | 80.0% | 99.9% | 0.7% | 0.043 |
| logistic_regression | 92.5% | 0.925 | 82.8% | 99.5% | 89.7% | 96.5% | 94.1% | 0.268 |
| k_nearest_neighbours | 84.2% | 0.842 | 88.5% | 83.9% | 86.8% | 97.1% | 64.7% | 1.424 |
| random_forest | 92.5% | 0.926 | 92.2% | 97.1% | 91.2% | 98.3% | 83.8% | 10.212 |

## Accuracy per held-out session

| Method | haseeb/s1 | haseeb/s2 | haseeb/s3 |
|---|---|---|---|
| rules | 74.9% | 74.4% | 80.0% |
| logistic_regression | 96.4% | 89.2% | 84.1% |
| k_nearest_neighbours | 85.0% | 81.9% | 84.1% |
| random_forest | 92.9% | 93.8% | 90.0% |

Mean over sessions (each session counts equally): rules 76.4% · logistic_regression 89.9% · k_nearest_neighbours 83.7% · random_forest 92.2%

## Accuracy per recording condition

| Method | far | normal | normal2 |
|---|---|---|---|
| rules | 78.5% | 76.1% | 72.4% |
| logistic_regression | 98.4% | 88.9% | 97.5% |
| k_nearest_neighbours | 87.0% | 85.6% | 77.1% |
| random_forest | 98.5% | 92.3% | 87.2% |

## Learning curve (logistic regression)

| Training frames | Accuracy |
|---|---|
| 50 | 81.3% |
| 100 | 89.8% |
| 250 | 89.0% |
| 500 | 91.8% |
| 1250 | 90.4% |
| 2500 | 90.3% |
| 5000 | 89.9% |

The rules returned a "Half Up" label on 1 of 7500 frames.

Figures: confusion_*.png, learning_curve.png. Raw numbers: metrics.json.
