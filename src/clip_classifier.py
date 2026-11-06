import torch
from transformers import CLIPModel, CLIPProcessor

class ClipClassifier:
    """Decides the class of each box by comparing the cropped image with text descriptions."""

    def __init__(self, model_id, class_texts, other_texts, device="cuda"):
        self.device = device
        self.num_classes = len(class_texts)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.model = CLIPModel.from_pretrained(model_id, dtype=torch.float16).to(device).eval()

        # One embedding per class: the average of its descriptions ("prompt ensemble")
        class_emb = torch.stack([self._embed_text(texts).mean(dim=0) for texts in class_texts])
        class_emb = class_emb / class_emb.norm(dim=-1, keepdim=True)
        other_emb = self._embed_text(other_texts)
        self.text_emb = torch.cat([class_emb, other_emb]) # rows: our classes first, then the "other" texts

    @torch.no_grad()
    def _embed_text(self, texts):
        inputs = self.processor(text=[f"a photo of {t}." for t in texts], return_tensors="pt", padding=True).to(self.device)
        emb = self.model.get_text_features(**inputs).pooler_output
        return emb / emb.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def classify(self, image, boxes, pad):
        """boxes: (N, 4) x1y1x2y2 in pixels. Returns class ids (-1 = other) and CLIP probabilities."""
        if len(boxes) == 0:
            return torch.zeros(0, dtype=torch.long), torch.zeros(0)
        W, H = image.size
        crops = []
        for x1, y1, x2, y2 in boxes.tolist():
            # A little context around the object helps CLIP recognize it
            pw, ph = (x2 - x1)*pad, (y2 - y1)*pad
            crops.append(image.crop((max(0, x1 - pw), max(0, y1 - ph), min(W, x2 + pw), min(H, y2 + ph))))

        pixel = self.processor(images=crops, return_tensors="pt")["pixel_values"].to(self.device, torch.float16)
        emb = self.model.get_image_features(pixel_values=pixel).pooler_output
        emb = emb / emb.norm(dim=-1, keepdim=True)
        probs = (100*emb @ self.text_emb.T).softmax(dim=-1).float().cpu() #100 = CLIP's temperature

        probs, ids = probs.max(dim=1)
        ids[ids >= self.num_classes] = -1 # an "other" text won
        return ids, probs