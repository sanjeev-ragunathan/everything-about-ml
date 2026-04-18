'''
BERT
- fine tuned for sentiment analysis
- take a Pre-Trained BERT model and teach it to classify movie reviews as positive or negative

Structure:
1. Load Pre-Trained BERT
2. Swap / add output layer for our task
3. Fine-Tune on Sentiment Analysis Dataset

NOTE: We'll take the model from Hugging Face.
'''

# ruff: noqa: E402 # to ignore "imports not on top of the file" warning


# Issue:
# Hugging Face models expects tokenized input - not raw text.
# Hence Tokenize first, then feed to model


# TOKENIZATION - introduction
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
tokens = tokenizer("Hello, how are you?", return_tensors='pt')
print(tokens)
# {'input_ids': tensor([[ 101, 7592, 1010, 2129, 2024, 2017, 1029,  102]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1]])}
# input_ids: token ids
# token_type_ids: this is for next sentence prediction step of BERT's pre-training (first sentence = 0, second sentence = 1), we have only one sentence here so all 0s
# attention_mask: which tokens are relevant (1 for real tokens, 0 for padding)



# FINETUNING

# LOAD IMDB dataset
from datasets import load_dataset
dataset = load_dataset('imdb')
print(dataset)
print(dataset['train'][0])

# TOKENIZE the dataset
# Use a small subset — full IMDB on CPU would take hours
small_train = dataset["train"].select(range(1000))
small_test = dataset["test"].select(range(500))

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_data = small_train.map(tokenize, batched=True)
test_data = small_test.map(tokenize, batched=True)

# Set format for PyTorch
train_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])
test_data.set_format("torch", columns=["input_ids", "attention_mask", "label"])

# LOAD PRE-TRAINED BERT - for sequence classification
from transformers import BertForSequenceClassification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# TRAINING LOOP
from torch.utils.data import DataLoader
import torch
train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

for epoch in range(3):
    for batch in train_loader:
        optimizer.zero_grad()
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["label"]
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")



# EVALUATION
model.eval()
total, correct = 0, 0
with torch.no_grad():
    for batch in DataLoader(test_data, batch_size=16):
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"]
        )
        preds = outputs.logits.argmax(dim=1)
        total += batch["label"].size(0)
        correct += (preds == batch["label"]).sum().item()
print(f"Test Accuracy: {correct / total:.4f}")
