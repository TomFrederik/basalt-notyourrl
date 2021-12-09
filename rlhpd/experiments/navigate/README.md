Test preference learning with MineRLNavigateDense-v0

```bash
export MINERL_DATA_ROOT=basalt-notyourrl/data
python3 -m minerl.data.download --environment MineRLNavigateDenseVectorObf-v0
```

Then
```bash
export PYTHONPATH="${PYTHONPATH}:../../"

python navigate_1_clips_from_demos.py

streamlit run navigate_2_inspect_clips.py

python navigate_3_train_reward.py

python navigate_4_train_policy.py
```