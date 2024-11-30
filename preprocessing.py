from datasets import DatasetDict
from datasets import ClassLabel

class DataPreprocessor:
    @staticmethod
    def remove_label(dataset, label_name):

        """Deletes the specified label from the dataset"""

        label_names = dataset['train'].features['label'].names
        label_index = label_names.index(label_name)
        return dataset['train'].filter(lambda example: example['label'] != label_index)

    @staticmethod
    def remap_label(dataset, old_label, new_label):

        """Renames labels in the dataset"""
        
        label_names = dataset['train'].features['label'].names
        new_label_names = [name if name != old_label else new_label for name in label_names]
        new_label_feature = ClassLabel(names=new_label_names)
        dataset['train'] = dataset['train'].map(lambda example: {'label': new_label if example['label'] == old_label else example['label']})
        dataset['train'] = dataset['train'].cast_column('label', new_label_feature)
        return dataset
