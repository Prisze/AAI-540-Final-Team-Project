import os, json, numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import joblib

def model_fn(model_dir):
    # Load the serialized "state" dict
    model_path = os.path.join(model_dir, "model.pkl")
    state = joblib.load(model_path)
    # Normalize
    feats = np.asarray(state["features"])
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    state["_unit_feats"] = feats / norms
    return state

def input_fn(request_body, content_type):
    if content_type.startswith("application/json"):
        return json.loads(request_body)
    elif content_type.startswith("text/csv"):
        parts = [p.strip() for p in request_body.split(",")]
        is_numeric = True
        vals = []
        for p in parts:
            try:
                vals.append(float(p))
            except Exception:
                is_numeric = False
                break
        if is_numeric and len(vals) > 0:
            return {"features": vals}
        else:
            return {"track_id": parts[0]}
    else:
        raise ValueError(f"Unsupported content type: {content_type}")

def _topn_from_vector(vec, state, n=10):
    vec = np.asarray(vec, dtype=float).reshape(1, -1)
    # Align dims if needed
    if state["_unit_feats"].shape[1] != vec.shape[1]:
        raise ValueError(f"Feature length mismatch: got {vec.shape[1]}, expected {state['_unit_feats'].shape[1]}")
    # Normalize input
    vnorm = np.linalg.norm(vec, axis=1, keepdims=True)
    vnorm[vnorm == 0] = 1.0
    uvec = vec / vnorm
    sims = (uvec @ state["_unit_feats"].T).ravel()
    top = np.argsort(sims)[::-1][:int(n)]
    ids = state["track_ids"]
    return [{"track_id": str(ids[i]), "similarity": float(sims[i])} for i in top]

def predict_fn(data, state):
    n = int(data.get("n", 10))
    if "track_id" in data:
        # Find the track and compute similarities against all tracks, excluding self when present
        tid = data["track_id"]
        ids = state["track_ids"]
        
        idx = np.where(ids == tid)[0]
        if len(idx) == 0:
            try:
                idx = np.where(ids == int(tid))[0]
            except Exception:
                pass
        if len(idx) == 0:
            try:
                idx = np.where(ids.astype(str) == str(tid))[0]
            except Exception:
                pass
        if len(idx) == 0:
            if "features" in data:
                return {"recommendations": _topn_from_vector(data["features"], state, n=n)}
            raise ValueError(f"track_id '{tid}' not found in index.")
        i = int(idx[0])
        vec = state["features"][i]
        recs = _topn_from_vector(vec, state, n=n+1)  # +1 so we can drop the item itself
        # Remove self if present at rank 1
        self_id = str(ids[i])
        filtered = [r for r in recs if r["track_id"] != self_id]
        return {"recommendations": filtered[:n]}
    elif "features" in data:
        return {"recommendations": _topn_from_vector(data["features"], state, n=n)}
    else:
        raise ValueError("Provide either 'track_id' or 'features' in the request payload.")

def output_fn(prediction, accept):
    body = json.dumps(prediction)
    ctype = "application/json"
    return body, ctype