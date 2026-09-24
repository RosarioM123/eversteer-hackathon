"""Submit E1 blend to the practice board (free diagnostic lane) with its model .pkl."""
from everestapi import EverestAPI
import json
api = EverestAPI()  # reads EIQ_API_KEY env
res = api.submit_validation_diagnostics(
    model_id="4041bfff-954c-4fb8-b058-e83eedeeea58",  # slot A - primary ensemble
    predictions="/home/hatch/workspace/everesteer_research/final/final_blend.csv",
    model_pkl="/home/hatch/workspace/everesteer_research/final/e1_predict.pkl",
    model_pkl_python_version="3.12",
    wait=True, timeout=900, poll_interval=5.0,
)
print(json.dumps(res, indent=1))
