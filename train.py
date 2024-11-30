from transformers import Trainer, TrainingArguments
import numpy as np
from datasets import Dataset
from sklearn.metrics import accuracy_score

class EmotionRecognitionTrainer:
    def __init__(self, model, train_data, val_data, label2id, id2label, training_args=None):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data
        self.label2id = label2id
        self.id2label = id2label
        self.training_args = training_args

    def compute_metrics(self, eval_pred):
        """
        Computes metrics for the model.
        """
        predictions = np.argmax(eval_pred.predictions, axis=1)
        return {"accuracy": accuracy_score(eval_pred.label_ids, predictions)}

    def train(self):
        """
        Starts the training process using the Trainer class
        """
        args = TrainingArguments(
            "emotion_model",
            **self.training_args
        )
        
        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_data,
            eval_dataset=self.val_data,
            compute_metrics=self.compute_metrics,
        )
        trainer.train()
        return trainer
