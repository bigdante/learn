import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Load the BERT model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')
model = model.to('cpu') # move the model to GPU, if available

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# Define the training data
inputs = [tokenizer.encode(text) for text in ["This is the first sample.", "This is the second sample."]]
labels = [0, 1] # class labels for the training examples

# Convert the training data to tensors
inputs = torch.tensor(inputs)
labels = torch.tensor(labels)

# Create a TensorDataset from the training data
dataset = TensorDataset(inputs, labels)

# Create a DataLoader from the TensorDataset
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Train the model
for epoch in range(1, 5):
    for batch in dataloader:
        input_ids, label = batch
        optimizer.zero_grad()
        loss, logits = model(input_ids.to('cpu'), labels=label.to('cpu'))
        loss.backward()
        optimizer.step()
    print("Epoch:", epoch, "Loss:", loss.item())

# Save the trained model
model.save_pretrained('trained_model')
