import torch
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
from Fact_Extraction.data_load import TripleExtractionDataset
from config_loader import config
from logging_loader import logging

logger = logging.getLogger("Fact_Extraction_Model")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "t5-small"
tokenizer = T5Tokenizer.from_pretrained(model_name,legacy=True)
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

logger.info("Beginning Data Extraction")

# Dataset returns: input_ids, attention_mask, labels
train_dataset = TripleExtractionDataset(config["paths"]["Dataset"]["Model_Input"]["Train_Data"], tokenizer)
val_dataset = TripleExtractionDataset(config["paths"]["Dataset"]["Model_Input"]["Validation_Data"], tokenizer)
test_dataset = TripleExtractionDataset(config["paths"]["Dataset"]["Model_Input"]["Test_Data"], tokenizer)

logger.info("Dataset Returned")

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=4)
test_loader = DataLoader(test_dataset, batch_size=4)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)

num_epochs = 30

for epoch in range(num_epochs):

    model.train()
    total_train_loss = 0

    for batch in train_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item()

    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            total_val_loss += outputs.loss.item()

    logger.info(
        f"Epoch {epoch} | "
        f"Train Loss: {total_train_loss:.4f} | "
        f"Val Loss: {total_val_loss:.4f}"
    )

model.eval()
total_test_loss = 0

with torch.no_grad():
    for batch in test_loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        total_test_loss += outputs.loss.item()

logger.info(f"Final Test Loss: {total_test_loss:.4f}")
