from transformers import AutoModelForImageClassification

class EmotionRecognitionModel:
    def __init__(self, model_checkpoint: str, label2id: dict, id2label: dict):
        self.model = AutoModelForImageClassification.from_pretrained(
            model_checkpoint,
            label2id=label2id,
            id2label=id2label,
            ignore_mismatched_sizes=True
        )

    def get_model(self):
        return self.model