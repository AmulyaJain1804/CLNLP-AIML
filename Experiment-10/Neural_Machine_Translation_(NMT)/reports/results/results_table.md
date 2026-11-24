# Test-set results

Held-out test split (see `docs/task_notes.md`). 95% bootstrap CIs over 1000 resamples; `*` = significantly better than the baseline, `†` = significantly worse (paired bootstrap, one-sided p < 0.05).

## chrF++ (test)

| Direction | Transformer baseline | + Back-translation | NLLB-200 fine-tuned |
|---|---:|---:|---:|
| en-as | 28.88 <sub>[28.2, 29.6]</sub> | 27.50† <sub>[26.8, 28.2]</sub> | **41.17* <sub>[40.2, 42.0]</sub>** |
| as-en | 32.31 <sub>[31.6, 33.0]</sub> | 32.54 <sub>[31.9, 33.3]</sub> | **49.14* <sub>[48.2, 50.2]</sub>** |
| en-mni | 40.70 <sub>[39.8, 41.7]</sub> | 40.76 <sub>[39.8, 41.7]</sub> | **55.72* <sub>[54.6, 56.9]</sub>** |
| mni-en | 42.38 <sub>[41.4, 43.4]</sub> | 41.75† <sub>[40.8, 42.8]</sub> | **61.47* <sub>[60.3, 62.6]</sub>** |
| en-lus | 39.96 <sub>[39.3, 40.6]</sub> | 39.21† <sub>[38.6, 39.8]</sub> | **46.90* <sub>[46.1, 47.7]</sub>** |
| lus-en | 36.95 <sub>[36.2, 37.6]</sub> | 36.16† <sub>[35.5, 36.8]</sub> | **50.22* <sub>[49.3, 51.1]</sub>** |
| en-bodo | 50.99 <sub>[49.7, 52.2]</sub> | 51.74* <sub>[50.4, 53.0]</sub> | **55.25* <sub>[54.2, 56.3]</sub>** |
| bodo-en | 47.08 <sub>[45.9, 48.3]</sub> | 48.45* <sub>[47.3, 49.7]</sub> | **59.07* <sub>[57.8, 60.4]</sub>** |
| **Average** | 39.91 | 39.76 | 52.37 |

## BLEU (test)

| Direction | Transformer baseline | + Back-translation | NLLB-200 fine-tuned |
|---|---:|---:|---:|
| en-as | 9.28 <sub>[8.6, 9.9]</sub> | 8.85† <sub>[8.2, 9.5]</sub> | **16.65* <sub>[15.8, 17.5]</sub>** |
| as-en | 14.43 <sub>[13.7, 15.3]</sub> | 14.28 <sub>[13.6, 15.1]</sub> | **30.04* <sub>[28.9, 31.2]</sub>** |
| en-mni | 12.38 <sub>[11.4, 13.3]</sub> | 12.81 <sub>[11.9, 13.7]</sub> | **23.86* <sub>[22.6, 25.2]</sub>** |
| mni-en | 18.63 <sub>[17.4, 19.8]</sub> | 17.93† <sub>[16.8, 19.0]</sub> | **38.71* <sub>[37.3, 40.2]</sub>** |
| en-lus | 19.75 <sub>[18.9, 20.5]</sub> | 19.11† <sub>[18.3, 19.8]</sub> | **24.74* <sub>[23.8, 25.6]</sub>** |
| lus-en | 18.69 <sub>[17.9, 19.5]</sub> | 17.61† <sub>[16.9, 18.3]</sub> | **32.33* <sub>[31.3, 33.4]</sub>** |
| en-bodo | 22.55 <sub>[21.0, 24.1]</sub> | **23.49* <sub>[21.7, 25.1]</sub>** | 23.03 <sub>[21.6, 24.4]</sub> |
| bodo-en | 22.25 <sub>[20.8, 23.7]</sub> | 23.31* <sub>[21.9, 24.8]</sub> | **38.48* <sub>[36.6, 40.3]</sub>** |
| **Average** | 17.24 | 17.17 | 28.48 |

## COMET (test)

| Direction | Transformer baseline | + Back-translation | NLLB-200 fine-tuned |
|---|---:|---:|---:|
| en-as | 65.10 <sub>[64.5, 65.6]</sub> | 61.44† <sub>[60.7, 62.1]</sub> | **79.07* <sub>[78.6, 79.5]</sub>** |
| as-en | 62.87 <sub>[62.3, 63.4]</sub> | 63.31* <sub>[62.7, 63.8]</sub> | **78.70* <sub>[78.2, 79.3]</sub>** |
| en-mni | 69.64 <sub>[69.0, 70.2]</sub> | 69.25† <sub>[68.5, 69.9]</sub> | **75.58* <sub>[75.0, 76.1]</sub>** |
| mni-en | 66.92 <sub>[66.1, 67.8]</sub> | 65.63† <sub>[64.8, 66.5]</sub> | **80.11* <sub>[79.4, 80.8]</sub>** |
| en-lus | 66.75 <sub>[66.3, 67.2]</sub> | 66.11† <sub>[65.6, 66.6]</sub> | **70.00* <sub>[69.5, 70.4]</sub>** |
| lus-en | 56.56 <sub>[56.0, 57.1]</sub> | 56.39 <sub>[55.9, 57.0]</sub> | **68.46* <sub>[67.9, 69.1]</sub>** |
| en-bodo | 62.55 <sub>[61.9, 63.1]</sub> | 62.36 <sub>[61.7, 62.9]</sub> | **64.72* <sub>[64.2, 65.3]</sub>** |
| bodo-en | 66.31 <sub>[65.4, 67.2]</sub> | 68.28* <sub>[67.4, 69.2]</sub> | **78.01* <sub>[77.2, 78.8]</sub>** |
| **Average** | 64.59 | 64.10 | 74.33 |

## NLLB-200 zero-shot vs fine-tuned (dev chrF++, greedy subset / beam full)

| Direction | Zero-shot (dev subset, greedy) | Fine-tuned (dev, beam) |
|---|---:|---:|
| en-as | 34.48 | 41.40 |
| as-en | 42.89 | 49.41 |
| en-mni | 40.73 | 55.88 |
| mni-en | 52.31 | 61.57 |
| en-lus | 38.25 | 46.36 |
| lus-en | 41.55 | 49.95 |
| en-bodo | 12.29 | 55.47 |
| bodo-en | 18.38 | 59.44 |

## Learning curve (baseline, test chrF++)

| Direction | Fraction | Train pairs | chrF++ | 95% CI |
|---|---:|---:|---:|---|
| en-as | 10% | 4852 | 18.90 | [18.4, 19.4] |
| en-as | 25% | 12130 | 22.09 | [21.6, 22.6] |
| en-as | 50% | 24260 | 25.68 | [25.1, 26.3] |
| en-as | 100% | 48521 | 28.88 | [28.2, 29.6] |
| as-en | 10% | 4852 | 22.25 | [21.8, 22.7] |
| as-en | 25% | 12130 | 26.10 | [25.5, 26.7] |
| as-en | 50% | 24260 | 30.34 | [29.7, 31.0] |
| as-en | 100% | 48521 | 32.31 | [31.6, 33.0] |

## Metric signatures

- **chrf++**: `nrefs:1|case:mixed|eff:yes|nc:6|nw:2|space:no|version:2.6.0`
- **bleu**: `nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0`
- **COMET**: `Unbabel/wmt22-comet-da`, system score = mean segment score x100
