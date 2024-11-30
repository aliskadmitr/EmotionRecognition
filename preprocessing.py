from datasets import ClassLabel

class DataPreprocessor:
    @staticmethod
    def remove_label(dataset, label_name, split_name='train', label_column='label'):

        """Deletes the specified label from the dataset"""

        label_names = dataset[split_name].features[label_column].names
        label_index = label_names.index(label_name)
        return dataset[split_name].filter(lambda example: example[label_column] != label_index)

    @staticmethod
    def remap_label(dataset, old_label, new_label, split_name='train', label_column='label'):

        """Renames labels in the dataset"""
        
        label_names = dataset[split_name].features[label_column].names
        new_label_names = [name if name != old_label else new_label for name in label_names]
        new_label_feature = ClassLabel(names=new_label_names)
        
        dataset[split_name] = dataset[split_name].map(
            lambda example: {label_column: new_label if example[label_column] == old_label else example[label_column]}
        )
        dataset[split_name] = dataset[split_name].cast_column(label_column, new_label_feature)
        
        return dataset
