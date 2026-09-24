"""Submit the excl20 rebuild to the practice board (free diagnostic lane) with the
required model .pkl artifact."""
from everestapi import EverestAPI
import json
api = EverestAPI()  # reads EIQ_API_KEY env
res = api.submit_validation_diagnostics(
    model_id="b3762a99-8fc6-4b2a-8b56-efe625324e73",  # slot C - experimental
    predictions="/home/hatch/workspace/everesteer_research/final/final_excl20_neut20.csv",
    model_pkl="/home/hatch/workspace/everesteer_research/final/excl20_predict.pkl",
    model_pkl_python_version="3.12",
    wait=True, timeout=900, poll_interval=5.0,
)
print(json.dumps(res, indent=1)[:3000])
