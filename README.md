# Adaptive Interventions with User-Defined Goals for Health Behavior Change

This repository contains all code required to reproduce the expirments for the following publication:

> Aishwarya Mandyam*, Matthew Jörke*, William Denton, Barbara E. Engelhardt, Emma Brunskill. Adaptive Interventions with User-Defined Goals for Health Behavior Change. , *Conference on Health, Inference, and Learning (CHIL) 2024*. 


### Instructions

1. `pip install -r requirements.txt`
2. Download `Codes & Data/Data_April21.zip` from <https://osf.io/9av87/?view_only=8bb9282111c24f81a19c2237e7d7eba3>. Unzip and copy `StepUp Data/pptdata.csv` to `models/gym/pptdata.csv`.
3. Run `python run_heartsteps.py`
4. Run `python run_gym.py` and `python run_gym.py --no-decay`
5. Run `python generate_figures.py`. The figures should appear in `plots`.