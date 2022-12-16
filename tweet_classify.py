import pandas as pd
import numpy as np
import torch
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from collections import defaultdict
from sklearn.metrics import classification_report
from tqdm import tqdm
import pdb
from utils import preprocess, preprocess_w_rationale, masked_cross_entropy
import torch.nn.functional as F


DATASET = load_dataset('hatexplain')
MAX_LEN = 128
BATCH_SIZE = 16
EPOCHS = 5
CLASS_NAMES = [0,1,2]
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
# train_data = concatenate_datasets([DATASET['train'], DATASET['validation']])
train_data = DATASET['train']
test_data = DATASET['test']
rationale_flag = False
lambda_att = 10

class TweetDataset(Dataset):

    def __init__(self, tweets, labels, tokenizer, max_len, rationales = None):
        self.tweet = tweets
        self.target = labels
        self.rationales = rationales
        
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.tweet)

    def __getitem__(self, item):
        tweet = str(self.tweet[item])
        label = int(self.target[item])
        encoding = self.tokenizer.encode_plus(
            tweet,
            max_length = self.max_len,
            pad_to_max_length = True,
            add_special_tokens = True,
            return_token_type_ids = False,
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        if self.rationales is not None:
            rationale = self.rationales[item]
            r_size = len(rationale)
            rationale = np.pad(rationale, (0, self.max_len-r_size), 'constant', constant_values=0.0)
            
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'rationales': torch.tensor(rationale, dtype=torch.float),
                'targets': torch.tensor(label, dtype=torch.long)
            }

        else:
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'targets': torch.tensor(label, dtype=torch.long)
            }


class TweetClassifier(torch.nn.Module):
    def __init__(self, n_classes):
        super(TweetClassifier, self).__init__()
        self.bert = model
        self.dropout = torch.nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(self.bert.config.hidden_size, n_classes)
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, input_ids, attention_masks):
        
        _, pooled_out, hidden = self.bert(input_ids = input_ids, attention_mask = attention_masks, return_dict = False, output_hidden_states = True)
        out = self.dropout(pooled_out)
        out = self.out(out)

        last_hidden = hidden[-1]

        return self.softmax(out), last_hidden


def train_epoch(model, data_loader, loss_fn, optimizer, device, scheduler, n_examples):
    model = model.train()

    losses = []
    correct_predictions = 0
    softmax_fn = torch.nn.Softmax(dim=-1)
    
    pbar = tqdm(data_loader)
    for d in pbar:
        input_ids = d['input_ids'].to(device)
        attention_mask = d['attention_mask'].to(device)
        answers = d['targets'].to(device)

        outs, last_hidden = model(
            input_ids = input_ids,
            attention_masks = attention_mask
        )
        # -> outs.shape = [batch size, #classes]

        _, preds = torch.max(outs,dim=1)
        loss = loss_fn(outs, answers)
        correct_predictions += torch.sum(preds == answers)

        if rationale_flag:
            rationales = d['rationales'].to(device)
            last_hidden = torch.nn.functional.softmax(last_hidden, dim=-1)
            last_hidden = torch.mean(last_hidden, dim=-1)
            # last_hidden = F.normalize(last_hidden, dim=-1)
            # r_loss = masked_cross_entropy(last_hidden, rationales, attention_mask)
            r_loss = loss_fn(last_hidden*attention_mask, rationales*attention_mask)
            loss =+ lambda_att * r_loss
        
        losses.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm = 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    pbar.close()
    
    return correct_predictions.double()/n_examples, np.mean(losses)

def eval_model(model, data_loader, loss_fn, device, n_examples):
    model = model.eval()
    
    losses = []
    correct_predictions = 0
    
    with torch.no_grad():
        for d in data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            answers = d["targets"].to(device)
            
            outs, last_hidden = model(
                        input_ids = input_ids,
                        attention_masks = attention_mask)
            
            _, preds = torch.max(outs, dim=1)
            
            loss = loss_fn(outs, answers)
            correct_predictions += torch.sum(preds == answers)
            losses.append(loss.item())
            
    return correct_predictions.double() / n_examples , np.mean(losses)

def get_predictions(model, data_loader):
    model = model.eval()
    predictions = []
    pred_probs = []
    sentiments = []

    with torch.no_grad():
        for d in data_loader:

            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            answers = d["targets"].to(device)
                
            outs = model(
                        input_ids = input_ids,
                        attention_masks = attention_mask)
                
            _, preds = torch.max(outs, dim=1)
            
            predictions.extend(preds)
            pred_probs.extend(outs)
            sentiments.extend(answers)

    predictions = torch.stack(predictions).cpu()
    pred_probs = torch.stack(pred_probs).cpu()
    sentiments = torch.stack(sentiments).cpu()

    return predictions, pred_probs, sentiments

if __name__ == '__main__':
    torch.manual_seed(0)

    history = defaultdict(list)
    best_accuracy = 0
    
    if rationale_flag:
        train = preprocess_w_rationale(train_data)
        test = preprocess_w_rationale(test_data)

        train_data_loader = torch.utils.data.DataLoader(
            TweetDataset(tweets = train["text"].to_numpy(),
            labels = train["label"].to_numpy(),
            rationales = train["rationale"].to_numpy(),
            tokenizer = tokenizer,
            max_len = MAX_LEN
            ),
            batch_size = BATCH_SIZE,
            shuffle = True)

        test_data_loader = torch.utils.data.DataLoader(
            TweetDataset(tweets = test["text"].to_numpy(),
            labels = test["label"].to_numpy(),
            rationales = test["rationale"].to_numpy(),
            tokenizer = tokenizer,
            max_len = MAX_LEN
            ),
            batch_size = BATCH_SIZE
            )

    else:
        train = preprocess(train_data)
        test = preprocess(test_data)

        train_data_loader = torch.utils.data.DataLoader(
            TweetDataset(tweets = train["text"].to_numpy(),
            labels = train["label"].to_numpy(),
            tokenizer = tokenizer,
            max_len = MAX_LEN
            ),
            batch_size = BATCH_SIZE,
            shuffle = True)

        test_data_loader = torch.utils.data.DataLoader(
            TweetDataset(tweets = test["text"].to_numpy(),
            labels = test["label"].to_numpy(),
            tokenizer = tokenizer,
            max_len = MAX_LEN
            ),
            batch_size = BATCH_SIZE
            )
    
    model = TweetClassifier(len(CLASS_NAMES))
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = torch.nn.CrossEntropyLoss().to(device)

    for epoch in range(EPOCHS):
        print(f'Epoch {epoch+1}/{EPOCHS}')
        print('-' * 15)

        train_acc, train_loss = train_epoch(
                                    model,
                                    train_data_loader,
                                    loss_fn,
                                    optimizer,
                                    device,
                                    scheduler,
                                    len(train))
        
        print(f'Train Loss {train_loss} Train Accuracy {train_acc}')
    
        test_acc, test_loss = eval_model(
                            model,
                            test_data_loader,
                            loss_fn,
                            device,
                            len(test))
    
        print(f'Test Loss {test_loss} Test Accuracy {test_acc}')
        history['test_acc'].append(test_acc)
        history['test_loss'].append(test_loss)
        
        if (test_acc > best_accuracy):
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = test_acc

    # test_preds, test_pred_probs, test_sents = get_predictions(model, test_data_loader)
    # print(classification_report(test_sents, test_preds, target_names = CLASS_NAMES))