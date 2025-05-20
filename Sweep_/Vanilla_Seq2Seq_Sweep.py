# Sweep file for the vanilla Seq2Seq Models #

# Importing the necessary libraries needed
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms.functional as Fn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch import Trainer
from torch.utils.data import DataLoader, Subset, Dataset
from lightning.pytorch.loggers import WandbLogger
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import wandb
# Wandb login
wandb.login(key = "5ef7c4bbfa350a2ffd3c198cb9289f544e3a0910")

# Data preparation
# Loading the dataset
df_train = pd.read_csv('/home/joel/DA6401_DL/DA6401_A03/ta_lexicons/ta.translit.sampled.train.tsv', sep='\t',  header=None, names=["native","latin","count"])
df_test = pd.read_csv('/home/joel/DA6401_DL/DA6401_A03/ta_lexicons/ta.translit.sampled.test.tsv', sep='\t',  header=None, names=["native","latin","count"])
df_val = pd.read_csv('/home/joel/DA6401_DL/DA6401_A03/ta_lexicons/ta.translit.sampled.dev.tsv', sep='\t',  header=None, names=["native","latin","count"])


# Show first few rows
print(df_train.head())

# Preparing the dataset for the model to fit #
class Dataset_Tamil(Dataset):
    def __init__(self, dataframe, build_vocab=True, input_token_index=None, output_token_index=None,
                 max_enc_seq_len=0, max_dec_seq_len=0):
        
        # Input variables
        self.input_df = dataframe
        self.input_words = []
        self.output_words = []
        # Characters of the language
        self.input_characters = set()
        self.output_characters = set()

        # Iterating thorough the rows
        for _, row in self.input_df.iterrows():
            input_word = str(row["latin"])
            output_word = "\t" + str(row["native"]) + "\n"
            self.input_words.append(input_word)
            self.output_words.append(output_word)
        
        if build_vocab:
            self.build_vocab()
        else:
            # Token index for sequence building
            self.input_token_index = input_token_index
            self.output_token_index = output_token_index
            # Heuristics lengths for the encoder decoder
            self.max_enc_seq_len = max_enc_seq_len
            self.max_dec_seq_len = max_dec_seq_len

        # Finding the encoder/decoder tokens 
        self.total_encoder_tokens = len(self.input_token_index)
        self.total_decoder_tokens = len(self.output_token_index)

    def build_vocab(self):
        # Building the vocabulary
        self.input_characters = sorted(set(" ".join(self.input_words)))
        self.output_characters = sorted(set(" ".join(self.output_words)))
        # Adding the padding character if not present
        if " " not in self.input_characters:
            self.input_characters.append(" ")
        if " " not in self.output_characters:
            self.output_characters.append(" ")

        # Fitting/Finding the necessary values from training data
        self.input_token_index = {char: i for i, char in enumerate(self.input_characters)}
        self.output_token_index = {char: i for i, char in enumerate(self.output_characters)}

        self.max_enc_seq_len = max(len(txt) for txt in self.input_words)
        self.max_dec_seq_len = max(len(txt) for txt in self.output_words)

    def __len__(self):
        return len(self.input_words)
    
    def __getitem__(self, index):
        input_word = self.input_words[index]
        output_word = self.output_words[index]

        # Finding the input for each stages of the network
        encoder_input = np.zeros((self.max_enc_seq_len, self.total_encoder_tokens), dtype=np.float32)
        decoder_input = np.zeros((self.max_dec_seq_len, self.total_decoder_tokens), dtype=np.float32)
        decoder_output = np.zeros((self.max_dec_seq_len, self.total_decoder_tokens), dtype=np.float32)

        for t, char in enumerate(input_word):
            if char in self.input_token_index:
                encoder_input[t, self.input_token_index[char]] = 1.0
        for t in range(len(input_word), self.max_enc_seq_len):
            encoder_input[t, self.input_token_index[" "]] = 1.0

        for t, char in enumerate(output_word):
            if char in self.output_token_index:
                decoder_input[t, self.output_token_index[char]] = 1.0
                if t > 0:
                    decoder_output[t - 1, self.output_token_index[char]] = 1.0
        # Fill remaining positions with space character
        for t in range(len(output_word), self.max_dec_seq_len):
            decoder_input[t, self.output_token_index[" "]] = 1.0

        for t in range(len(output_word) - 1, self.max_dec_seq_len):
            decoder_output[t, self.output_token_index[" "]] = 1.0

        return (
            torch.from_numpy(encoder_input),
            torch.from_numpy(decoder_input),
            torch.from_numpy(decoder_output)
        )
    
# Model classes definitions #
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3, cell_type="RNN", num_layers=1, bi_directional=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.cell_type = cell_type.upper()
        self.dropout = dropout
        self.num_layers = num_layers

        if self.cell_type == 'LSTM':
            self.enc = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=self.dropout, num_layers=self.num_layers, bidirectional=bi_directional)
        elif self.cell_type == 'GRU':
            self.enc = nn.GRU(input_size, hidden_size, batch_first=True, dropout=self.dropout, num_layers=self.num_layers, bidirectional=bi_directional)
        else:
            self.enc = nn.RNN(input_size, hidden_size, batch_first=True, dropout=self.dropout, num_layers=self.num_layers, bidirectional=bi_directional)

    def forward(self, x):
        if self.cell_type == "LSTM":
            hidden, (hn, cn) = self.enc(x)
            return hidden, (hn, cn)
        else:
            hidden, out = self.enc(x)
            return hidden, out
        

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.3, cell_type='RNN', num_layers=1):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.cell_type = cell_type.upper()
        self.dropout = dropout
        self.num_layers = num_layers

        if self.cell_type == 'LSTM':
            self.dec = nn.LSTM(input_size, hidden_size, batch_first=True, dropout=self.dropout, num_layers=self.num_layers)
        elif self.cell_type == 'GRU':
            self.dec = nn.GRU(input_size, hidden_size, batch_first=True, dropout=self.dropout, num_layers=self.num_layers)
        else:
            self.dec = nn.RNN(input_size, hidden_size, batch_first=True, dropout=self.dropout, num_layers=self.num_layers)

    def forward(self, x, states):
        if type(states) == tuple:
            hidden, (hn, cn) = self.dec(x, states)
            return hidden, (hn, cn)
        else:
            hidden, out = self.dec(x, states)
            return hidden, out
        
# Helper function
def combine_directions(hidden):
    layers = []
    for i in range(0, hidden.size(0), 2):  
        fwd = hidden[i]
        bwd = hidden[i + 1]
        combined = torch.cat((fwd, bwd), dim=-1)  
        layers.append(combined)
    return torch.stack(layers)  


class Seq2Seq(nn.Module):
    def __init__(self, input_token_index, output_token_index, max_dec_seq_len, embedding_dim,hidden_size_enc, bi_directional,
            nature="train", enc_cell="LSTM", dec_cell="LSTM", num_layers=1,dropout=0.2, device="cpu"):
        super(Seq2Seq, self).__init__()
        self.input_index_token = input_token_index
        self.output_index_token = output_token_index
        self.max_dec_seq_len = max_dec_seq_len
        self.nature = nature
        self.enc_cell_type = enc_cell.upper()
        self.dec_cell_type = dec_cell.upper()
        self.num_layers= num_layers
        self.bi_directional = bi_directional
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = (1 + int(self.bi_directional == True))*hidden_size_enc
        self.embedding = nn.Linear(in_features=len(self.input_index_token), out_features=embedding_dim)
        self.embedding_act = nn.Tanh()
        self.encoder = Encoder(input_size=embedding_dim, hidden_size=hidden_size_enc, dropout=dropout, cell_type=enc_cell, num_layers=num_layers, bi_directional=self.bi_directional).to(device)
        self.decoder = Decoder(input_size=len(self.output_index_token), hidden_size=self.hidden_size_dec, dropout=dropout, cell_type=dec_cell, num_layers=num_layers).to(device)
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.fc = nn.Linear(in_features=self.hidden_size_dec, out_features=len(output_token_index))

    def forward(self, batch):
        ENC_IN, DEC_IN, DEC_OUT = batch
        ENC_IN = ENC_IN.to(self.device)
        DEC_IN = DEC_IN.to(self.device)

        batch_size = ENC_IN.size(0)
        input_embedding = self.embedding_act(self.embedding(ENC_IN))
        hidden_enc, states_enc = self.encoder(input_embedding)

        if self.bi_directional == True:
            if self.enc_cell_type == "LSTM":
                (h,c) = states_enc
                states_enc = (combine_directions(h), combine_directions(c))
            else:
                states_enc = combine_directions(states_enc)

        # Teacher forcing mode #    
        # Making the states correctly formatted
        if self.dec_cell_type == "LSTM": 
            if isinstance(states_enc, tuple):
                states_dec = states_enc
            else:
                h = torch.zeros(self.num_layers, batch_size, self.decoder.hidden_size, device=self.device)
                c = states_enc
                states_dec = (h, c)
        else:
            if isinstance(states_enc, tuple):
                states_dec = states_enc[1]
            else:
                states_dec = states_enc

        # Decoder gives the outputs batchwise
        decoder_outputs, _ = self.decoder(DEC_IN, states_dec)  
        logits = self.fc(decoder_outputs)                   
        return logits

    def predict_greedy(self, batch):
        # Greedy force outputs #
        ENC_IN, DEC_IN, DEC_OUT = batch
        ENC_IN = ENC_IN.to(self.device)
        DEC_IN = DEC_IN.to(self.device)

        batch_size = ENC_IN.size(0)
        input_embedding = self.embedding_act(self.embedding(ENC_IN))
        hidden_enc, states_enc = self.encoder(input_embedding)

        if self.bi_directional == True:
            if self.enc_cell_type == "LSTM":
                (h,c) = states_enc
                states_enc = (combine_directions(h), combine_directions(c))
            else:
                states_enc = combine_directions(states_enc)
            
        # Final matrix
        final_out = torch.zeros(batch_size, self.max_dec_seq_len, len(self.output_index_token), device=self.device)

        # Initial decoder input (with start token)
        in_ = torch.zeros(batch_size, 1, len(self.output_index_token), device=self.device)
        in_[:, 0, 0] = 1.0
        # Making the states correctly formatted
        if self.dec_cell_type == "LSTM":
            if isinstance(states_enc, tuple):
                states_dec = states_enc
            else:
                h = torch.zeros(self.num_layers, batch_size, self.decoder.hidden_size, device=self.device)
                c = states_enc
                states_dec = (h, c)
        else:
            if isinstance(states_enc, tuple):
                states_dec = states_enc[1]
            else:
                states_dec = states_enc

        # Output to input
        for t in range(self.max_dec_seq_len):
            out_step, states_dec = self.decoder(in_, states_dec)  
            logits_step = self.fc(out_step.squeeze(1))            
            final_out[:, t, :] = logits_step

            # Greedy argmax for next input
            top1 = torch.argmax(logits_step, dim=1)               
            in_ = torch.zeros(batch_size, 1, len(self.output_index_token), device=self.device)
            in_[torch.arange(batch_size), 0, top1] = 1.0

        return final_out

# Fucntion for validation of the model # 
def validate_seq2seq(model, val_loader, device, val_type = "greedy", beam_width=None):
    model.eval()
    total_loss = 0.0
    correct_chars = 0
    total_chars = 0
    correct_words = 0
    total_words = 0
    loss_fn = nn.CrossEntropyLoss(ignore_index=2)

    with torch.no_grad():
        for batch in val_loader:
            ENC_IN, DEC_IN, DEC_OUT = batch
            ENC_IN = ENC_IN.to(device)
            DEC_IN = DEC_IN.to(device)
            DEC_OUT = DEC_OUT.to(device)

            # Forward pass
            decoder_output = model(batch)

            # Compute loss
            vocab_size = decoder_output.size(-1)
            decoder_output = decoder_output.view(-1, vocab_size)
            decoder_target_indices = DEC_OUT.argmax(dim=-1).view(-1)

            loss = loss_fn(decoder_output, decoder_target_indices)
            total_loss += loss.item()

            # Character-wise accuracy
            if val_type == "greedy":
                decoder_output = model.predict_greedy(batch)
            else:
                decoder_output = model.predict_beam_search(batch, beam_width=beam_width)

            #print(decoder_output.shape)
            pred_tokens = decoder_output.argmax(dim=2)
            true_tokens = DEC_OUT.argmax(dim=2)
            #print(pred_tokens.shape)
            #print(true_tokens.shape)
            
            mask = true_tokens != 2  # Ignore PAD tokens
            correct_chars += (pred_tokens[mask] == true_tokens[mask]).sum().item()
            total_chars += mask.sum().item()

            mask = true_tokens != 2  # Ignore PAD tokens
            #print(mask.shape)
            total_words += decoder_output.shape[0]
            #print(pred_tokens[mask].shape)
            chk_words = (mask.int() - (pred_tokens == true_tokens).int())
            chk_words[mask == False] = 0
            correct_words += (chk_words.sum(dim = 1) == 0).sum().item()

    avg_loss = total_loss / len(val_loader)
    accuracy = correct_chars / total_chars if total_chars > 0 else 0.0
    word_acc = correct_words / total_words if total_words > 0 else 0.0
    return avg_loss, accuracy, word_acc


# Trainloop
def train_seq2seq(model, train_loader, val_loader, optimizer, num_epochs, device, beam_sizes = [3,5], run=None):
    loss_fn = nn.CrossEntropyLoss(ignore_index=2)  # 2 is the padding index
    max_val_char_acc = 0
    max_val_word_acc = 0
    print("Training of the model has started...")
    counter = 0
    patience = 7
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        tqdm_loader = tqdm(train_loader, desc=f"Epoch : {epoch + 1} ", ncols=100)

        for batch in tqdm_loader:
            ENC_IN, DEC_IN, DEC_OUT = batch
            ENC_IN = ENC_IN.to(device)
            DEC_IN = DEC_IN.to(device)
            DEC_OUT = DEC_OUT.to(device)
            # Move to device
            decoder_output = model(batch)

            # Reshape for loss
            decoder_output = decoder_output.view(-1, decoder_output.size(-1))
            decoder_target_indices = DEC_OUT.argmax(dim=-1).view(-1)

            loss = loss_fn(decoder_output, decoder_target_indices)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            tqdm_loader.set_postfix({"Train Loss": loss.item()})

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_loss:.4f}")

        val_loss, val_acc, val_word_acc = validate_seq2seq(model, val_loader, device)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val Word Acc: {val_word_acc:.4f}")

        if run is not None:
            run.log({"train_loss_epoch" : avg_loss, "val_loss_epoch" : val_loss, "val_char_acc" : val_acc, "val_word_acc" : val_word_acc})

        if val_word_acc > max_val_word_acc or val_acc > max_val_char_acc:
            max_val_char_acc = val_acc
            max_val_word_acc = val_word_acc
            counter = 0
        else:
            counter += 1

        if counter > patience:
            break

    if run is not None:
        run.summary["max_val_char_acc"] = max_val_char_acc
        run.summary["max_val_word_acc"] = max_val_word_acc
        
        

# Function to run the sweep
def run_sweep(config = None):
    torch.cuda.empty_cache()
    run = wandb.init(config=config)
    config = wandb.config
    run_name = f"bs_{config.batch_size}_enc_{config.enc_cell_type}_dec_{config.dec_cell_type}_bi_{config.bi_directional}_hidden_size_{config.hidden_size_enc}"
    run.name = run_name

    # Loading the datasets and dataloaders
    train_dataset = Dataset_Tamil(df_train)
    val_dataset = Dataset_Tamil(df_val, build_vocab=False, input_token_index=train_dataset.input_token_index, 
                                output_token_index=train_dataset.output_token_index, max_enc_seq_len=train_dataset.max_enc_seq_len,
                                max_dec_seq_len=train_dataset.max_dec_seq_len)
    test_dataset = Dataset_Tamil(df_test, build_vocab=False, input_token_index=train_dataset.input_token_index, 
                                output_token_index=train_dataset.output_token_index, max_enc_seq_len=train_dataset.max_enc_seq_len,
                                max_dec_seq_len=train_dataset.max_dec_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Seq2Seq(input_token_index=train_dataset.input_token_index, output_token_index=train_dataset.output_token_index, max_dec_seq_len=train_dataset.max_dec_seq_len,
                    embedding_dim=config.embedding_dim, hidden_size_enc=config.hidden_size_enc, bi_directional=config.bi_directional, enc_cell=config.enc_cell_type, dec_cell=config.dec_cell_type, 
                    num_layers=config.num_layers, dropout=config.dropout_rnn, device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    train_seq2seq(model, train_loader, val_loader, optimizer, num_epochs=config.epochs, device=device, run=run)


sweep_config = {
    "name" : "Vanilla Seq2Seq",
    "method" : "random",
    "metric" : {
        "name" : "val_char_acc",
        "goal" : "maximize"
    },
    "parameters" : {
        "learning_rate" : {"values" : [0.001]},
        "dropout_rnn" : {"values" : [0, 0.1, 0.2, 0.4]}, 
        "batch_size" : {"values" :  [64, 128, 256]},
        "epochs" : {"values" : [40]},
        "embedding_dim" : {"values" : [16, 32, 64, 128, 256]},
        "num_layers" : {"values" : [1, 2, 3, 5]},
        "hidden_size_enc" : {"values" : [32, 64, 128, 256]},
        "enc_cell_type" : {"values" : ["RNN", "LSTM", "GRU"]},
        "dec_cell_type" : {"values" : ["RNN", "LSTM", "GRU"]},
        "bi_directional" : {"values" : [True, False]}
    }
}

sweep_id = wandb.sweep(sweep=sweep_config, project="Vanilla_RNN", entity="A3_DA6401_DL")
wandb.agent(sweep_id, function=run_sweep,project="Vanilla_RNN", entity="A3_DA6401_DL")
