import os
import sys
from glob import glob
import heapq
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
from torch.optim.lr_scheduler import CyclicLR
import Transformer_Model as Model

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(model, source, target, optimizer, criterion, scheduler = None):
  model.train()
  optimizer.zero_grad()
  source, target = source.to(device), target.to(device)
  dec_input = target[:, :-1]
  dec_target = target[:, 1:]
  preds = model(source, dec_input)
  preds = F.log_softmax(preds, dim=-1)
  loss = criterion(preds, dec_target)
  loss.backward()
  nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  optimizer.step()
  if(scheduler != None):
    scheduler.step()
  return loss.item()

@torch.no_grad()
def test(model, source, target, criterion):
  model.eval()
  source, target = source.to(device), target.to(device)
  dec_input = target[:, :-1]
  dec_target = target[:, 1:]
  preds = model(source, dec_input)
  preds = F.log_softmax(preds, dim=-1)
  loss = criterion(preds, dec_target)
  return loss.item()

@torch.no_grad()
def generate(model, source, target_start_token_idx):
  source = source.to(device)
  bs = source.shape[0]
  enc = model.encoder(source)
  dec_input = torch.ones((bs, 1), dtype=torch.int32) * target_start_token_idx
  dec_input = dec_input.to(device)
  for i in range(model.target_maxlen - 1):
    dec_out = model.decoder(enc, dec_input, 1)
    logits = model.classifier(dec_out)
    #logits = F.log_softmax(logits, dim=-1)

    logits = torch.argmax(logits, dim=-1)
    last_logit = torch.unsqueeze(logits[:, -1], axis=1)
    dec_input = torch.cat((dec_input, last_logit), axis=-1)
  return dec_input


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
  vectorizer = VectorizeChar(200)
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
  

def build_dataset(batch_size):
  max_target_len = 200
  dir = "/home/rnakaha2/documents/speech/LJSpeech-1.1"
  wavs = glob("{}/**/*.wav".format(dir), recursive=True)
  id_to_text = {}
  with open(os.path.join(dir, "metadata.csv"), encoding="utf-8") as f:
    for line in f:
      id = line.strip().split("|")[0]
      text = line.strip().split("|")[2]
      id_to_text[id] = text
    
  raw_data = get_data(wavs, id_to_text, max_target_len)
  vectorizer = VectorizeChar(max_target_len)
  #num_classes = len(vectorizer.get_vocabulary())

  split = int(len(raw_data) * 0.90)
  train_data = raw_data[:split]
  test_data = raw_data[split:]

  trn_dataset = AudioTextDataset(train_data)
  test_dataset = AudioTextDataset(test_data)
  trn_dl = DataLoader(trn_dataset, batch_size=batch_size, shuffle=True)
  test_dl = DataLoader(test_dataset, batch_size=8, shuffle=True)
  return trn_dl, test_dl

def display_outputs(model, source, target, idx_to_token, num_display, target_start_token_idx, target_end_token_idx):
  preds = generate(model, source, target_start_token_idx)
  preds = preds.cpu().detach().numpy()
  print()
  for i in range(num_display):
    target_text = "".join([idx_to_token[_] for _ in target[i, :]])
    prediction = ""
    for idx in preds[i, :]:
        prediction += idx_to_token[idx]
        if idx == target_end_token_idx:
          break
    print(f"target:     {target_text.replace('-','')}")
    print(f"prediction: {prediction}\n")


# def ModelLoss(pred, target):
#   lossfn = nn.CTCLoss(blank=0)
#   pred_length = torch.full((pred.size(0),), pred.size(1), dtype=torch.long, device=device)
#   mask = target == 3
#   indices = mask.nonzero(as_tuple=False)
#   target_length = torch.full((target.size(0),), target.size(0)-1 , dtype=torch.long, device=device)
#   target_length[indices[:, 0]] = indices[:, 1] + 1
#   pred = torch.permute(pred, (1, 0, 2))
#   loss = lossfn(pred, target, pred_length, target_length)
#   return loss

def ModelLoss(pred, target):
  lossfn = nn.CrossEntropyLoss(ignore_index=0)  # Use CrossEntropyLoss
  pred = pred.transpose(1, 2)  # Transpose predictions to match the shape expected by CrossEntropyLoss
  loss = lossfn(pred, target)
  return loss

def train_model(model, lr, n_epochs, trn_dl, test_dl, print_freq, step_size):
  print("SpeechToText Model\tlr: ", lr,"\tTotal Epochs: ", n_epochs, "\tNumber of Data: ", len(trn_dl))
  optimizer = optim.Adam(model.parameters(), lr=lr)
  criterion = ModelLoss
  scheduler = None
  vectorizer = VectorizeChar(200)
  #scheduler = CyclicLR(optimizer, base_lr=0.000001, max_lr=lr, step_size_up=step_size, mode='exp_range', cycle_momentum=False)
  idx_to_token = vectorizer.get_vocabulary()

  for epoch in range(n_epochs):
    progress = 0
    audio, text = None, None
    for audio, text in trn_dl:
      loss = train(model, audio, text, optimizer, criterion, scheduler)
      progress+=1
      if((progress) % print_freq == 0):
        sys.stdout.write("\rEpoch: {}\t{}%\tlr: {}\tLoss: {}".format(epoch+1, int(progress/len(trn_dl)*100), lr ,round(loss, 5)))
        sys.stdout.flush()
    display_outputs(model, audio, text, idx_to_token, num_display=2, target_start_token_idx=2, target_end_token_idx=3)
    val_avg_loss = 0
    for audio, text in test_dl:
      val_avg_loss += test(model, audio, text, criterion)
    print("\nEpoch: {}\tValidation Loss: {}".format(epoch+1, round(val_avg_loss/len(test_dl),5)))

    
  torch.save(model.state_dict(), 'model_beta.pth')

def initialize_weights(model):
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Apply Xavier/Glorot initialization to linear layers
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv2d) or isinstance(module, nn.Conv1d):
            # Apply Xavier/Glorot initialization to convolutional layers
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0.0)

def main():
  batch_size = 32
  trn_dl, test_dl = build_dataset(batch_size)
  model = Model.Transformer(num_hid=200, num_head=2, num_feed_forward=450, num_layers_enc=6, num_layers_dec=1).to(device)
  initialize_weights(model)

  n_epoch = 35
  max_lr = 0.00269155
  print_freq = 5
  step_size = len(trn_dl)/2

  train_model(model, max_lr, n_epoch, trn_dl, test_dl, print_freq, step_size)

if __name__ == "__main__":
  main()