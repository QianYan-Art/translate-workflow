import base64
import io
import json
from typing import List, Dict
import requests

def _render_pages(path: str, dpi: int, pages_spec: str | None) -> List[bytes]:
    import fitz
    doc = fitz.open(path)
    indices = []
    if pages_spec:
        parts = [p.strip() for p in str(pages_spec).split(',') if p.strip()]
        for t in parts:
            if '-' in t:
                a, b = t.split('-', 1)
                try:
                    s = int(a)
                    e = int(b)
                except Exception:
                    continue
                if s <= e:
                    rng = range(s, e + 1)
                else:
                    rng = range(s, e - 1, -1)
                for i in rng:
                    if 1 <= i <= len(doc):
                        indices.append(i - 1)
            else:
                try:
                    i = int(t)
                except Exception:
                    continue
                if 1 <= i <= len(doc):
                    indices.append(i - 1)
    else:
        indices = list(range(len(doc)))
    pixmaps = []
    for i in indices:
        page = doc.load_page(i)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pm = page.get_pixmap(matrix=mat, alpha=False)
        pixmaps.append(pm)
    images = []
    for pm in pixmaps:
        images.append(pm.tobytes("png"))
    return images

def _post_json(url: str, key: str, payload: Dict) -> Dict:
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}" if key else "",
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    resp.raise_for_status()
    return resp.json()

def _encode_images(images: List[bytes]) -> List[str]:
    return [base64.b64encode(b).decode("ascii") for b in images]

def recognize_pdf(path: str, cfg: Dict) -> str:
    mode = str(cfg.get("mode") or "auto").lower()
    dpi = int(cfg.get("dpi") or 200)
    pages_spec = cfg.get("pages")
    images = _render_pages(path, dpi, pages_spec)
    if mode == "none":
        return "\n\n".join([""] * len(images))
    vlm_url = cfg.get("vlm_url") or ""
    vlm_key = cfg.get("vlm_key") or ""
    use_vlm = False
    if mode == "vlm":
        use_vlm = bool(vlm_url)
    else:
        use_vlm = bool(vlm_url)
    b64 = _encode_images(images)
    if use_vlm:
        payload = {
            "images": b64,
            "prompt": "Read the page images and output only the text as plain text.",
        }
        try:
            data = _post_json(vlm_url, vlm_key, payload)
            texts = data.get("texts")
            if isinstance(texts, list) and texts:
                out = []
                for t in texts:
                    out.append(t or "")
                return "\n\n".join(out)
            t = data.get("text") or ""
            return t
        except Exception:
            pass
    return "\n\n".join([""] * len(images))