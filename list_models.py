"""List the agent's models (read-only) to pick a model_id for the diagnostic upload."""
from everestapi import EverestAPI
import json
api = EverestAPI()  # reads EIQ_API_KEY env
res = api.get_models()
print(json.dumps(res, indent=1)[:2000])
