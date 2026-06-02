# Salinity ML Lab

Everything runs from **this folder** (`Salinity 2`).

## Files

| File | Purpose |
|------|---------|
| `Data.xlsx` | Your dataset (also in `data/Data.xlsx`) |
| `app.py` | Streamlit UI |
| `Salinity_2.py` | Command-line experiments |
| `run_ui.bat` | Double-click to open the UI |
| `run_cli.bat` | Double-click for a CLI run |

## Setup (once)

```bash
pip install -r requirements.txt
```

## Run

- **UI:** double-click `run_ui.bat` or run `streamlit run app.py` from this folder.
- **CLI:** double-click `run_cli.bat` or run `python Salinity_2.py --runs 30`.

## UI tabs

| Tab | What you control |
|-----|------------------|
| **Data** | Preview rows and class balance |
| **Features** | Drop columns *or* use only selected columns |
| **Hyperparameters** | Full sklearn settings per algorithm (save/load JSON) |
| **Run** | Results, feature importance chart, JSON export |

**Sidebar:** resampling, SMOTE neighbors, hold-out vs stratified CV, auto-tune decision threshold for recall/F1.

Open this folder in Visual Studio and set **Working Directory** to `.` if you debug from the IDE.
