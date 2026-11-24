# SOURCES — data provenance and license log (owner: M1)

* **Task:** WMT26 Shared Task on Low-Resource Indic Language Translation — https://www2.statmt.org/wmt26/indic-mt-task.html
* **Release:** official 2026 training data, Google Drive folder https://drive.google.com/drive/folders/1TsaINvXwb5GbiEbEkj2OgcK-1wfrWESG
  (downloaded with `python -m src.data.download`; SHA-256 checksums in
  `data/raw/_original/checksums.json`).
* **Official 2026 test set / 2025 gold test set:** downloaded (`test2026.zip`, `test2025.zip`)
  but **password protected** (released only to registered participants) — not used.
* **License / usage:** the task page states no explicit license. We treat the data as
  *research use only, for the purpose of the shared task*, do not redistribute it, and
  cite the task. Original corpus creators are listed by the organisers in the
  'Language Resource Contributors' sheet linked from the task page.
* **Our splits:** dev/test are held out from the training release with a fixed seed
  (42); dev = min(5%, 1000), test = min(5%, 2000).

| Pair | Language | Cat. | Original file | Rows | Pairs kept | Train | Dev | Test | SHA-256 (first 12) |
|---|---|---|---|---:|---:|---:|---:|---:|---|
| en-as | Assamese | I | `Category I/English-Assamese Training Data 2026.csv` | 54000 | 54000 | 51000 | 1000 | 2000 | `1bc62c568fac` |
| en-mni | Manipuri (Bengali script) | I | `Category I/English-Manipuri Training Data 2026.xlsx` | 23687 | 23687 | 21503 | 1000 | 1184 | `4b9a052eb5a2` |
| en-lus | Mizo | I | `Category I/English-Mizo Traning Data 2026.xlsx` | 50000 | 50000 | 47000 | 1000 | 2000 | `8bd00c418002` |
| en-bodo | Bodo | II | `Category II/English-Bodo Training Data 2026.xlsx` | 15215 | 15214 | 13694 | 760 | 760 | `956541ab40e7` |
| en-kha | Khasi | I | `Category I/English-Khasi Training Data 2026.xlsx` | 26000 | 26000 | 23700 | 1000 | 1300 | `34a41d27c491` |
| en-mni-Mtei | Manipuri (Meitei Mayek script) | I | `Category I/English-Meitei-Mayek Transining Data 2026.xlsx` | 17000 | 17000 | 15300 | 850 | 850 | `d74e7580c6fa` |
| en-njz | Nyishi | I | `Category I/English-Nyishi Training Data 2026.xlsx` | 60000 | 60000 | 57000 | 1000 | 2000 | `92ed0c91ed39` |
| en-mjw | Karbi | II | `Category II/English - Karbi Training Data 2026.xlsx` | 1000 | 1000 | 900 | 50 | 50 | `f193ea834ef0` |
| en-trp | Kokborok | II | `Category II/English-Kokborok Training Data 2026.xlsx` | 2269 | 2269 | 2043 | 113 | 113 | `fff327941ad2` |
| en-nag | Nagamese | II | `Category II/English-Nagamese  Training Data 2026.xlsx` | 7205 | 2000 | 1800 | 100 | 100 | `4d2f87d23a54` |
| en-tgj | Tagin | II | `Category II/English-Tagin Training Data 2026.xlsx` | 5343 | 5343 | 4809 | 267 | 267 | `94af5a25110a` |

Rows = data rows in the spreadsheet (header excluded); 'pairs kept' drops rows
where both cells are empty (spreadsheet padding — notably ~5.2k rows in Nagamese).
