import os
import wandb
import pprint
from glob import glob
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import Transformer_Model as Model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  
os.environ['WANDB_NOTEBOOK_NAME'] = 'Sweep_wandb.ipynb'
wandb.login()

num_trails = 1
sweep_config = {'method': 'random'}
metric = {'name': 'loss','goal': 'minimize'}
sweep_config['metric'] = metric

# parameters_dict = {
#     'ff_dim': {'values':[300, 350, 400, 450, 500]},
#     'hidden_dim':{'values':[150, 200 , 250, 300]}
#   }

# sweep_config['parameters'] = parameters_dict

# parameters_dict.update({'epochs': {'value': 10}})

# parameters_dict.update({
#   'learning_rate': {'distribution': 'uniform','min': 0.00000001,'max': 0.001},
#   'beta1': {'distribution': 'uniform','min': 0.6,'max': 0.97},
#   'beta2': {'distribution': 'uniform','min': 0.7,'max': 1},
#   'batch_size': {'distribution': 'q_log_uniform_values','q': 8,'min': 8,'max': 64}
#   })
parameters_dict = {
    'ff_dim': {'values':[500]},
    'hidden_dim':{'values':[250]}
  }

sweep_config['parameters'] = parameters_dict

parameters_dict.update({'epochs': {'value': 100}})

parameters_dict.update({
  'learning_rate': {'value': 0.0004},
  'beta1': {'value': 0.77},
  'beta2': {'value':  0.94},
  'batch_size': {'value': 32}
  })

pprint.pprint(sweep_config)

def get_data(wavs, id_to_text, maxlen=50):
  """ returns mapping of audio paths and transcription texts """
  data = []
  for w in wavs:
    id = w.split("/")[-1].split(".")[0]
    if len(id_to_text[id]) < maxlen:
      data.append({"audio": w, "text": id_to_text[id]})
  return data

class VectorizeChar:
  def __init__(self, max_len=50):
    self.vocab = (
      ["-", "#", "<", ">"]
      + [chr(i + 96) for i in range(1, 27)]
      + [" ", ".", ",", "?"]
    )
    self.max_len = max_len
    self.char_to_idx = {}
    for i, ch in enumerate(self.vocab):
      self.char_to_idx[ch] = i

  def __call__(self, text):
    text = text.lower()
    text = text[: self.max_len - 2]
    text = "<" + text + ">"
    pad_len = self.max_len - len(text)
    return [self.char_to_idx.get(ch, 1) for ch in text] + [0] * pad_len

  def get_vocabulary(self):
    return self.vocab
  

def path_2_audio(path):
  waveform, _ = torchaudio.load(path)
  audio = torch.squeeze(waveform, dim=0)
  stfts = torch.stft(audio, n_fft=256, hop_length=80, win_length=200, return_complex=True)
  x = torch.pow(torch.abs(stfts), 0.5)
  means = torch.mean(x, 1, keepdims=True)
  stddevs = torch.std(x, 1, keepdims=True)
  x = (x - means) / stddevs
  pad_len = 2754
  paddings = (0, pad_len, 0, 0)
  x = F.pad(x, paddings, 'constant',0)[:, :pad_len]
  return x

def create_text_ds(data):
  vectorizer = VectorizeChar(max_len=200)
  texts = [_["text"] for _ in data]
  text_ds = [vectorizer(t) for t in texts]
  return text_ds


class AudioTextDataset(Dataset):
  def __init__(self, raw_data):
    self.audio = [_["audio"] for _ in raw_data]
    self.text = create_text_ds(raw_data)

  def __len__(self):
    return len(self.text)
  def __getitem__(self, i):
    audio = path_2_audio(self.audio[i])
    text = torch.tensor(self.text[i])
    return audio, text
  

def load_data():
  dir = "/home/rnakaha2/documents/speech/LJSpeech-1.1"
  wavs = glob("{}/**/*.wav".format(dir), recursive=True)
  id_to_text = {}
  with open(os.path.join(dir, "metadata.csv"), encoding="utf-8") as f:
    for line in f:
      id = line.strip().split("|")[0]
      text = line.strip().split("|")[2]
      id_to_text[id] = text
  raw_data = get_data(wavs, id_to_text, maxlen=200)
  return raw_data
raw_data = load_data()

def build_dataset(raw_data, batch_size):
  dataset = AudioTextDataset(raw_data)
  trn_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
  return trn_dl

def build_network(batch_size, hidden_dim, ff_dim):
  model = Model.Transformer(batch_size=batch_size, num_hid=hidden_dim, num_head=2, num_feed_forward=ff_dim, 
           num_layers_enc=4, num_layers_dec=1).to(device)
  return model

def generate(model, source, target_start_token_idx):
  source = source.to(device)
  bs = source.shape[0]
  enc = model.encoder(source)
  dec_input = torch.ones((bs, 1), dtype=torch.int32) * target_start_token_idx
  dec_input = dec_input.to(device)
  for i in range(model.target_maxlen - 1):
    dec_out = model.decoder(enc, dec_input, 1)
    logits = model.classifier(dec_out)
    logits = torch.argmax(logits, dim=-1)
    last_logit = torch.unsqueeze(logits[:, -1], axis=1)
    dec_input = torch.cat((dec_input, last_logit), axis=-1)
  return dec_input

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0.0)

def build_optimizer(network, learning_rate, beta1, beta2):
  optimizer = optim.AdamW(network.parameters(), lr=learning_rate, betas=(beta1, beta2))
  return optimizer

def network_loss(pred, target):
  lossfn = nn.CrossEntropyLoss(ignore_index=0)
  pred = pred.transpose(1, 2)
  loss = lossfn(pred, target)
  return loss

def train_epoch(network, loader, optimizer):
  cumu_loss = 0
  audio, text = None, None
  for _, (audio, text) in enumerate(loader):
    audio, text = audio.to(device), text.to(device)
    dec_input = text[:, :-1]
    dec_target = text[:, 1:]
    optimizer.zero_grad()
    network.train()
    preds = network(audio, dec_input)
    preds = F.log_softmax(preds, dim=-1)

    loss = network_loss(preds, dec_target)
    cumu_loss += loss.item()
    loss.backward()
    optimizer.step()

    wandb.log({"batch loss": loss.item()})
  vectorizer = VectorizeChar(max_len=200)
  idx_to_token = vectorizer.get_vocabulary()
  preds = generate(network, audio[0:2], 2)
  preds = preds.cpu().detach().numpy()
  target_text = "".join([idx_to_token[_] for _ in text[0, :]])
  prediction = ""
  for idx in preds[0, :]:
      prediction += idx_to_token[idx]
      if idx == 3:
        break
  target = target_text.replace('-','')
  return cumu_loss / len(loader), target, prediction


def train(config=None):
  with wandb.init(config=config):
    table = wandb.Table(columns=["Epoch", "Loss","Prediction", "Target"])
    config = wandb.config
    loader = build_dataset(raw_data, config.batch_size)
    network = build_network(config.batch_size, config.hidden_dim, config.ff_dim)
    initialize_weights(network)
    optimizer = build_optimizer(network, config.learning_rate, config.beta1, config.beta2)

    for epoch in range(config.epochs):
      avg_loss, target, prediction = train_epoch(network, loader, optimizer)
      wandb.log({"loss": avg_loss, "epoch": epoch})
      table.add_data(epoch, avg_loss, prediction, target)
    wandb.log({"result": table})
    torch.save(network.state_dict(), 'Pretrained_ASR_Model.pth')
    del network

sweep_id = wandb.sweep(sweep_config, project="ASR_Model_Sweep")
wandb.agent(sweep_id, train, count=num_trails)