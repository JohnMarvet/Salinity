@echo off
cd /d "%~dp0"
echo Running Salinity ML CLI (30 runs, random forest)...
python Salinity_2.py --runs 30 --algorithm random_forest
pause
