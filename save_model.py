import torch

class ModelSaver:
    @staticmethod
    def save(model, model_path: str):
        """Saves the model to the specified directory"""
        torch.save(model.state_dict(), model_path)
        print(f"Model save into {model_path}")
