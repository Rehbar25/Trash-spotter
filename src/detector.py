import torch 
from torchvision.ops import nms
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

class GroundingDinoDetector:
    def __init__(self, model_id, phrases, device="cuda"):
        self.device = device
        self.phrases = phrases
        self.prompt = ". ".join(phrases) + "."
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device).eval()
        self.phrase_tokens = self._find_phrase_tokens()

    def _find_phrase_tokens(self):
        """For each phrase, the positions of its tokens inside the tokenised prompt."""
        offsets = self.processor.tokenizer(self.prompt, return_offsets_mapping=True)["offset_mapping"]
        result = []
        search_from = 0
        for phrase in self.phrases:
            start = self.prompt.index(phrase, search_from)
            end = start + len(phrase)
            search_from = end
            result.append([i for i, (a,b) in enumerate(offsets) if b > a and a >= start and b <= end])
        return result 

    @torch.no_grad()
    def detect(self, image, box_threshold, nms_iou):
        inputs = self.processor(images=image, text=self.prompt, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)


        # logits: one score per (query box, prompt token). Sigmoid turns them into 0...1.
        token_scores = outputs.logits[0].sigmoid() # (900, num_tokens)
        # Phrases score = best token score inside that phrase
        phrase_scores = torch.stack([token_scores[:, ids].max(dim=1).values for ids in self.phrase_tokens], dim=1)
        scores, phrase_ids = phrase_scores.max(dim=1) # best phrase for each box

        keep = scores > box_threshold
        boxes = self._to_pixels(outputs.pred_boxes[0][keep], image.size)
        scores, phrase_ids = scores[keep], phrase_ids[keep]

        # Class-agnostic NMS: overlapping boxes compete regardless of their phrase
        keep = nms(boxes, scores, nms_iou)
        return boxes[keep].cpu(), scores[keep].cpu(), phrase_ids[keep].cpu()

    @staticmethod
    def _to_pixels(boxes, image_size):
        """Model boxes are (center_x, center_y, width, height) in 0..1.
        Convert to pixel (x1, y1, x2, y2)."""

        w, h = image_size
        cx, cy, bw, bh = boxes.unbind(-1)
        return torch.stack([(cx - bw / 2)*w, (cy-bh / 2)*h, (cx + bw / 2)*w, (cy + bh / 2)*h], dim=-1)