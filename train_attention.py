# Train script for Attention Seq2Seq Model #

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
import argparse
wandb.login()

# Command line arguments (DEFAULT : Best Hyperparameters)
parser = argparse.ArgumentParser(description="Trainin a Attention Seq2Seq model !!!")
# adding the arguments #
parser.add_argument("-wp", '--wandb_project', type=str, default="projectname", help = "project name used in wandb dashboard to track experiments")
parser.add_argument("-we", "--wandb_entity", type=str, default="myname", help="Wandb enetity used to track the experiments")
parser.add_argument("-e", "--epochs", type=int, default=30, help = "Number of epochs to train the model")
parser.add_argument("-b", "--batch_size", type=int, default=256, help="Batch size used to train the network")
parser.add_argument("-dropout", "--dropout", type=float, default=0.2, help="Dropout to applu at the convolutional layers before activation")
parser.add_argument("-lr", "--learning_rate", type=float, default=0.001, help="learning rate used in the gradient update")
parser.add_argument("-bi_directional", "--bi_directional", type=bool, default=True, help="Whether to apply bi-directionality")
parser.add_argument("-num_layers", "--num_layers", type=int, default=2, help="Determine the number of layers")
parser.add_argument("-embed_dim", "--embed_dim", type=int, default=256, help="Determine the embedding dimension")
parser.add_argument("-hidden_size", "--hidden_size", type=int, default=128, help="Determine the hidden dimension")
parser.add_argument("-enc_cell", "--enc_cell", type=str, default="GRU", help="Determine the encoder cell type")
parser.add_argument("-dec_cell", "--dec_cell", type=str, default="RNN", help="Determine the decoder cell type")
parser.add_argument("-pre_trained", "--pre_trained", type=bool, default=True, help="Whether to use pretrained")

# parsing the arguments
args = parser.parse_args()

# Building the config
config = {
    "learning_rate" : args.learning_rate,
    "dropout_rnn" : args.dropout, 
    "batch_size" :  args.batch_size,
    "epochs" : args.epochs,
    "embedding_dim" : args.embed_dim,
    "num_layers" : args.num_layers,
    "hidden_size_enc" : args.hidden_size,
    "enc_cell_type" : args.enc_cell,
    "dec_cell_type" : args.dec_cell,
    "bi_directional" : args.bi_directional,
}

# Cache emptying and setting precision
torch.cuda.empty_cache()

# Data preparation
# Loading the dataset
df_train = pd.read_csv(os.path.join(os.path.abspath(""),'ta_lexicons/ta.translit.sampled.train.tsv'), sep='\t',  header=None, names=["native","latin","count"])
df_test = pd.read_csv(os.path.join(os.path.abspath(""),'ta_lexicons/ta.translit.sampled.test.tsv'), sep='\t',  header=None, names=["native","latin","count"])
df_val = pd.read_csv(os.path.join(os.path.abspath(""),'ta_lexicons/ta.translit.sampled.dev.tsv'), sep='\t',  header=None, names=["native","latin","count"])

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
        
class Attention_Mechanism(nn.Module):
    def __init__(self, hidden_dim, device="cpu"):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        # Creating the matrices for attention calculation
        self.V_att = nn.Parameter(torch.randn(size=(self.hidden_dim, 1), device=device)*0.1)
        self.U_att = nn.Parameter(torch.randn(size=(self.hidden_dim, self.hidden_dim), device=device)*0.1)
        self.W_att = nn.Parameter(torch.randn(size=(self.hidden_dim, self.hidden_dim), device=device)*0.1)

    def forward(self, st_1, c_j, mask):
        # Compute the attention scores and softmax
        """
        st_1 : input of size (bx1xd)
        c_j : input of size (bxLxd)
        """
        #print(st_1.shape, c_j.shape)
        inside = self.tanh(torch.matmul(c_j, self.W_att) + torch.matmul(st_1, self.U_att))
        #print(inside.shape)
        scores = torch.matmul(inside, self.V_att).squeeze(2)
        #print(scores.shape)
        scores[mask] = -torch.inf

        attention = self.softmax(scores)
        return attention
    
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
        if states == None:
            hidden, out = self.dec(x)
            return hidden, out
        elif type(states) == tuple:
            hidden, (hn, cn) = self.dec(x, states)
            return hidden, (hn, cn)
        else:
            hidden, out = self.dec(x, states)
            return hidden, out
        
class Attention_Seq2Seq(nn.Module):
    def __init__(self, input_token_index, output_token_index, max_dec_seq_len, embedding_dim,hidden_size_enc, bi_directional=False,
            nature="train", enc_cell="LSTM", dec_cell="LSTM", num_layers=1,dropout=0.2, device="cpu"):
        super().__init__()

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
        self.attention = Attention_Mechanism(hidden_dim=self.hidden_size_dec)
        self.decoder = Decoder(input_size=len(self.output_index_token)+self.hidden_size_dec, hidden_size=self.hidden_size_dec, dropout=dropout, cell_type=dec_cell, num_layers=num_layers).to(device)
        self.device = device
        self.loss_fn = nn.CrossEntropyLoss()
        self.fc = nn.Linear(in_features=self.hidden_size_dec, out_features=len(output_token_index))

    def forward(self, batch):
        ENC_IN, DEC_IN, DEC_OUT = batch
        ENC_IN = ENC_IN.to(self.device)
        DEC_IN = DEC_IN.to(self.device)

        batch_size = ENC_IN.size(0)
        input_embedding = self.embedding_act(self.embedding(ENC_IN))
        mask_ = torch.argmax(ENC_IN, 2) == 2
        hidden_enc, states_enc = self.encoder(input_embedding)

        # Final matrix
        final_out = torch.zeros(batch_size, self.max_dec_seq_len, len(self.output_index_token), device=self.device)

        # Initial decoder input (with start token)
        in_ = DEC_IN[:, 0:1, :].clone()
        for t in range(self.max_dec_seq_len):
            if t==0:
                out_step, states_dec = self.decoder(torch.cat((in_, hidden_enc[:,-1,:].unsqueeze(1)), dim=2), None)  
            else:
                # input for next input
                in_ = DEC_IN[:, t, :].unsqueeze(1).clone()
                att_scores = self.attention(out_step, hidden_enc, mask_)

                in_ = torch.cat((in_, torch.bmm(att_scores.unsqueeze(1), hidden_enc)), dim=2)
                # Output
                out_step, states_dec = self.decoder(in_, states_dec)  

            logits_step = self.fc(out_step.squeeze(1))          
            final_out[:, t, :] = logits_step
   
        return final_out
    
    def predict_greedy(self, batch):
        ENC_IN, DEC_IN, DEC_OUT = batch
        ENC_IN = ENC_IN.to(self.device)
        DEC_IN = DEC_IN.to(self.device)

        batch_size = ENC_IN.size(0)
        input_embedding = self.embedding_act(self.embedding(ENC_IN))
        mask_ = torch.argmax(ENC_IN, 2) == 2
        hidden_enc, states_enc = self.encoder(input_embedding)

        # Final matrix
        final_out = torch.zeros(batch_size, self.max_dec_seq_len, len(self.output_index_token), device=self.device)

        # Initial decoder input (with start token)
        in_ = torch.zeros(batch_size, 1, len(self.output_index_token), device=self.device)
        in_[:, 0, 0] = 1.0

        for t in range(self.max_dec_seq_len):
            if t==0:
                out_step, states_dec = self.decoder(torch.cat((in_, hidden_enc[:,-1,:].unsqueeze(1)), dim=2), None)  
            else:
                out_step, states_dec = self.decoder(in_, states_dec)  

            logits_step = self.fc(out_step.squeeze(1))            
            final_out[:, t, :] = logits_step

            # Greedy argmax for next input
            top1 = torch.argmax(logits_step, dim=1)               
            in_ = torch.zeros(batch_size, 1, len(self.output_index_token), device=self.device)
            in_[torch.arange(batch_size), 0, top1] = 1.0
            att_scores = self.attention(out_step, hidden_enc, mask_)

            in_ = torch.cat((in_, torch.bmm(att_scores.unsqueeze(1), hidden_enc)), dim=2)
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
        tqdm_progress = tqdm(val_loader, desc="Predicting ...")
        for batch in tqdm_progress:
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


def main_wrapper(config = None):
    torch.cuda.empty_cache()
    run = wandb.init(config=config, entity=args.wandb_entity, project=args.wandb_project)

    # Loading the datasets and dataloaders
    train_dataset = Dataset_Tamil(df_train)
    val_dataset = Dataset_Tamil(df_val, build_vocab=False, input_token_index=train_dataset.input_token_index, 
                                output_token_index=train_dataset.output_token_index, max_enc_seq_len=train_dataset.max_enc_seq_len,
                                max_dec_seq_len=train_dataset.max_dec_seq_len)
    test_dataset = Dataset_Tamil(df_test, build_vocab=False, input_token_index=train_dataset.input_token_index, 
                                output_token_index=train_dataset.output_token_index, max_enc_seq_len=train_dataset.max_enc_seq_len,
                                max_dec_seq_len=train_dataset.max_dec_seq_len)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Attention_Seq2Seq(input_token_index=train_dataset.input_token_index, output_token_index=train_dataset.output_token_index, max_dec_seq_len=train_dataset.max_dec_seq_len,
                    embedding_dim=config["embedding_dim"], hidden_size_enc=config["hidden_size_enc"], bi_directional=config["bi_directional"], enc_cell=config["enc_cell_type"], dec_cell=config["dec_cell_type"], 
                    num_layers=config["num_layers"], dropout=config["dropout_rnn"], device=device).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])

    if args.pre_trained:
        model.load_state_dict(torch.load(os.path.join(os.path.abspath(""), "pretrained/Attention_Best_model.pth"), weights_only=True))
    else:
        train_seq2seq(model, train_loader, val_loader, optimizer, num_epochs=config["epochs"], device=device, run=run)

    # Test data
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)
    _, test_char_acc, test_word_acc = validate_seq2seq(model, test_loader, device)

    if run is not None:
        run.summary["test_char_acc"] = test_char_acc
        run.summary["test_word_acc"] = test_word_acc

    print(f"Test Word Accuracy : {test_word_acc}")
    print(f"Test Char Acc : {test_char_acc}")

if __name__ == "__main__":
    main_wrapper(config=config)