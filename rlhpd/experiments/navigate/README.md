Test preference learning with MineRLNavigateDense-v0

```python
import minerl
minerl.data.download(data_dir, "MineRLNavigateDense-v0")
```

Then
```bash
export PYTHONPATH="${PYTHONPATH}:../../"
python navigate_1_clips_from_demos.py
```