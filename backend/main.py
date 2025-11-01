import io
import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
import clip  # from openai/CLIP
from labels import LABELS

app = FastAPI(title="Currency Detector API", version="1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Configurable thresholds (tune as needed) ----
BANKNOTE_PROB_THRESHOLD = 0.60   # gate: prob it's a banknote must be >= 0.60
TOP1_PROB_THRESHOLD     = 0.25   # classification: top class prob must be >= 0.25
TOP_MARGIN_THRESHOLD    = 0.05   # classification: (top1 - top2) must be >= 0.05

_device = None
_model = None
_preprocess = None
_text_tokens = None
_banknote_tokens = None

BANKNOTE_PROMPTS = [
    "a photo of a banknote",
    "not a banknote"
]

def _init_model():
    global _device, _model, _preprocess, _text_tokens, _banknote_tokens
    if _model is not None:
        return
    _device = "cuda" if torch.cuda.is_available() else "cpu"
    _model, _preprocess = clip.load("ViT-B/32", device=_device)
    _model.eval()
    _text_tokens = clip.tokenize(LABELS).to(_device)
    _banknote_tokens = clip.tokenize(BANKNOTE_PROMPTS).to(_device)

@app.on_event("startup")
async def startup_event():
    _init_model()

@app.get("/api/health")
async def health():
    return {"status": "ok"}

@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    _init_model()

    # --- Basic file validations ---
    if file.content_type is None or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Please upload an image file")

    try:
        content = await file.read()
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Invalid image data")

    with torch.no_grad():
        # Encode image once
        img = _preprocess(image).unsqueeze(0).to(_device)
        img_feat = _model.encode_image(img)
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # ---- Gate: Is this a banknote? ----
        bank_txt_feat = _model.encode_text(_banknote_tokens)
        bank_txt_feat = bank_txt_feat / bank_txt_feat.norm(dim=-1, keepdim=True)
        bank_logits = (100.0 * img_feat @ bank_txt_feat.T)
        bank_probs = bank_logits.softmax(dim=-1)[0]  # [prob_banknote, prob_not]
        prob_banknote = float(bank_probs[0].item())

        if prob_banknote < BANKNOTE_PROB_THRESHOLD:
            return {
                "is_currency": False,
                "reason": f"Low banknote confidence ({prob_banknote:.2f} < {BANKNOTE_PROB_THRESHOLD:.2f})",
            }

        # ---- Classify among known currency labels ----
        txt_feat = _model.encode_text(_text_tokens)
        txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)

        logits = (100.0 * img_feat @ txt_feat.T)
        probs = logits.softmax(dim=-1)[0]

        # top1
        top1 = probs.topk(1)
        best_score = float(top1.values.item())          # probability [0..1]
        best_idx = int(top1.indices.item())
        best_label = LABELS[best_idx]

        # top5
        top5 = probs.topk(5)
        top5_indices = [int(i) for i in top5.indices.tolist()]
        top5_scores = [float(s) for s in top5.values.tolist()]
        top5_labels = [LABELS[i] for i in top5_indices]

        # margin (top1 - top2)
        if probs.shape[0] >= 2:
            t2_val = float(probs.topk(2).values.tolist()[1])
            margin = best_score - t2_val
        else:
            margin = best_score  # degenerate case

        # ---- Sanity checks to reject non-currency images ----
        if best_score < TOP1_PROB_THRESHOLD:
            return {
                "is_currency": False,
                "reason": f"Low class confidence ({best_score:.2f} < {TOP1_PROB_THRESHOLD:.2f})",
                "banknote_confidence": prob_banknote,
            }

        if margin < TOP_MARGIN_THRESHOLD:
            return {
                "is_currency": False,
                "reason": f"Ambiguous prediction (margin {margin:.3f} < {TOP_MARGIN_THRESHOLD:.3f})",
                "banknote_confidence": prob_banknote,
            }

        return {
            "is_currency": True,
            "banknote_confidence": prob_banknote,
            "predicted": {"label": best_label, "score": best_score},
            "top5": [{"label": l, "score": s} for l, s in zip(top5_labels, top5_scores)]
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
