# DA6401 Assignment - 03 (AE21B105)
  This assignment involves training and tuning the hyperparameters of the seq2seq models for the dhakshina dataset. 

# Links to Report and Project
GitHub Link : https://github.com/AE21B105-JoelJ/DA6401_A03.git
WandB Link : https://api.wandb.ai/links/A3_DA6401_DL/t5ookbc5

# Packages Used
- torch
- argparse
- scikit-learn
- torchvision
- numpy
- pandas
- plotly
- seaborn
- lightning
- matplotlib
- tqdm

## Usage of the script
To run the code use the following code (The external parameters are defaulted to best accuracy got!!!), for the non-attention run the python script named train_vanilla.py and for the attention module run the train_attention.py. Use the following command to run (note: pre_trained is set as true to use existing model file, set it as false if want to train a new model)

```
python train_vanilla.py --wandb_project project_name --wandb_entity entity_name
```

If you want to train the model and save it in the pretrained folder and add the path to the script as required.

```
python train_vanilla.py --wandb_project project_name --wandb_entity entity_name --pre_trained False
```

the additional supported arguents are as follows (both types attention and non-attention)
- "--epochs" tells the Number of epochs to train the model
- "--batch_size" tells  the Batch size used to train the network
- "--dropout" tells the Dropout to applu at the convolutional layers before activation
- "--learning_rate" tells the learning rate used in the gradient update
- "--bi_directional" tells whether to apply bi-directionality
- "--num_layers" to determine the number of layers
- "--embed_dim" to determine the embedding dimension
- "--hidden_size" to determine the hidden dimension
- "--enc_cell" to determine the encoder cell type
- "--dec_cell" to determine the decoder cell type
- "--pre_trained" to tell whether to use pretrained

# Organization of the Repository
.
├── Attention_notebook.ipynb
├── Att_Weights
│   ├── C1.png
│   ├── C2.png
│   ├── C3.png
│   ├── FIG10.png
│   ├── FIG1.png
│   ├── FIG2.png
│   ├── FIG3.png
│   ├── FIG4.png
│   ├── FIG5.png
│   ├── FIG6.png
│   ├── FIG7.png
│   ├── FIG8.png
│   └── FIG9.png
├── Base_notebook.ipynb
├── Connectivity_Img
│   ├── C1.png
│   ├── C2.png
│   ├── C3.png
│   ├── C4.png
│   ├── C5.png
│   ├── FIG_10.png
│   ├── FIG_1.png
│   ├── FIG_2.png
│   ├── FIG_3.png
│   ├── FIG_4.png
│   ├── FIG_5.png
│   ├── FIG_6.png
│   ├── FIG_7.png
│   ├── FIG_8.png
│   └── FIG_9.png
├── DA6401_A3_AE21B105.pdf
├── LICENSE
├── predictions_attention
│   └── Attention_predictions_test.csv
├── predictions_vanilla
│   └── Vanilla_predictions_test.csv
├── pretrained
│   ├── Attention_Best_model.pth
│   ├── Attention_weights.pth
│   └── Vanilla_Best_model.pth
├── README.md
├── Report_Files
│   ├── ATT_WEIGHTS.ipynb
│   ├── REPORT_ATTENTION_TEST.ipynb
│   └── REPORT_VANILLA_TEST.ipynb
├── Sweep_
│   ├── Attention_Seq2Seq_Sweep.py
│   └── Vanilla_Seq2Seq_Sweep.py
├── ta_lexicons
│   ├── ta.translit.sampled.dev.tsv
│   ├── ta.translit.sampled.test.tsv
│   └── ta.translit.sampled.train.tsv
├── train_attention.py
├── train_vanilla.py
└── Vanilla_report_.ipynb
