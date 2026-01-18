# backend/app.py
import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
SVM_PATH = os.path.join(MODEL_DIR, "lyrics_genre_linearSVC_calibrated.pkl")
BUNDLE_PATH = os.path.join(MODEL_DIR, "final_sbert_svm_ensemble_w0.5.pkl")

print(">>> Starting backend. BASE_DIR:", BASE_DIR)
print(">>> Looking for models in:", MODEL_DIR)

# sanity ensure folder exists
if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)
    print(">>> Created model dir:", MODEL_DIR)

use_ensemble = False
svm = None
clf_sbert = None
sbert = None
weight = 1.0
classes = None
sbert_model_name = None

# Prefer final bundle if present
if os.path.exists(BUNDLE_PATH):
    print(f">>> Found bundle {BUNDLE_PATH}. Loading ensemble bundle...")
    try:
        bundle = joblib.load(BUNDLE_PATH)
    except Exception as e:
        raise RuntimeError(f"Failed loading bundle at {BUNDLE_PATH}: {e}")
    # expected keys: svm_with_proba, clf_sbert, sbert_model_name, ensemble_weight, classes
    try:
        svm = bundle["svm_with_proba"]
        clf_sbert = bundle["clf_sbert"]
        sbert_model_name = bundle.get("sbert_model_name", "all-MiniLM-L6-v2")
        weight = float(bundle.get("ensemble_weight", 0.5))
        classes = np.array(bundle["classes"])
        use_ensemble = True
        print(">>> Bundle loaded. Ensemble weight:", weight, "classes len:", len(classes))
    except Exception as e:
        raise RuntimeError(f"Bundle missing expected keys or invalid format: {e}")
else:
    print(f">>> Bundle not found at {BUNDLE_PATH}. Trying to load SVM fallback at {SVM_PATH}...")
    if not os.path.exists(SVM_PATH):
        raise FileNotFoundError(
            f"No model found. Please put one of these files in {MODEL_DIR}:\n"
            f" - lyrics_genre_linearSVC_calibrated.pkl  (SVM fallback)\n"
            f" - final_sbert_svm_ensemble_w0.5.pkl      (final ensemble bundle)\n"
        )
    try:
        svm = joblib.load(SVM_PATH)
        # try to extract classes robustly
        if hasattr(svm, "named_steps") and "calibratedclassifiercv" in svm.named_steps:
            classes = np.array(svm.named_steps['calibratedclassifiercv'].classes_)
        elif hasattr(svm, "named_steps") and "clf" in svm.named_steps:
            classes = np.array(svm.named_steps['clf'].classes_)
        else:
            # fall back to clf classes if present
            classes = np.array(getattr(svm, "classes_", []))
        print(">>> SVM loaded. Classes len:", len(classes))
    except Exception as e:
        raise RuntimeError(f"Failed loading SVM at {SVM_PATH}: {e}")

# Only import and load SBERT encoder if we will use ensemble (keeps startup light)
if use_ensemble:
    print(">>> Loading SBERT encoder:", sbert_model_name)
    try:
        from sentence_transformers import SentenceTransformer
        sbert = SentenceTransformer(sbert_model_name)
        print(">>> SBERT loaded.")
    except Exception as e:
        raise RuntimeError(f"Failed to load SBERT model '{sbert_model_name}': {e}")

# FastAPI app
app = FastAPI(title="Lyrics Genre Predictor")

# Allow simple CORS for dev; lock down in prod
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Inp(BaseModel):
    text: str

@app.get("/health")
def health():
    return {"status": "ok", "ensemble": use_ensemble, "classes": int(len(classes))}

@app.post("/predict")
def predict(inp: Inp):
    text = inp.text
    if not text or not text.strip():
        raise HTTPException(status_code=400, detail="Empty text")

    try:
        probs_svm = svm.predict_proba([text])[0]   # (n_classes,)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SVM predict_proba failed: {e}")

    if not use_ensemble:
        probs = probs_svm
    else:
        try:
            emb = sbert.encode([text])
            probs_sbert = clf_sbert.predict_proba(emb)[0]
            # reorder sbt classes to match svm class order if different
            if hasattr(clf_sbert, "classes_"):
                csbert = list(clf_sbert.classes_)
                order_map = [csbert.index(c) for c in classes]
                probs_sbert = probs_sbert[order_map]
            probs = weight * probs_svm + (1 - weight) * probs_sbert
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"SBERT path failed: {e}")

    idx = int(np.argmax(probs))
    return {
        "predicted_genre": str(classes[idx]),
        "confidence": float(probs[idx]),
        "scores": {str(classes[i]): float(probs[i]) for i in range(len(classes))}
    }
