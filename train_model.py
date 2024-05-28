from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset, DatasetDict
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def tokenize_function(examples, tokenizer):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    labels = p.label_ids
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    acc = accuracy_score(labels, preds)
    return {'accuracy': acc, 'f1': f1, 'precision': precision, 'recall': recall}

def train_model(df, model_name='bert-base-uncased', num_labels=3):
    # Prepare the data for training
    train_df = df.sample(frac=0.8, random_state=42)
    val_df = df.drop(train_df.index)

    # Convert to Hugging Face Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    dataset = DatasetDict({'train': train_dataset, 'validation': val_dataset})

    # Load the tokenizer and model
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Tokenize the dataset
    tokenized_datasets = dataset.map(lambda x: tokenize_function(x, tokenizer), batched=True)

    # Define the training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy='epoch',
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets['validation'],
        compute_metrics=compute_metrics,
    )

    # Fine-tune the model
    trainer.train()

    # Evaluate the model
    results = trainer.evaluate()
    print("Evaluation Results:", results)
    return model, results
