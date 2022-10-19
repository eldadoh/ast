import glob
import torch
import os, csv, argparse, wget
import torch, torchaudio, timm
import numpy as np
from torch.cuda.amp import autocast
from src.models import ASTModel
from loguru import logger
from src.utilities.util import get_model_checkpoint
from collections import Counter

get_model_checkpoint()


def make_features(wav_name, mel_bins, target_length=1024):
  waveform, sr = torchaudio.load(wav_name)

  fbank = torchaudio.compliance.kaldi.fbank(
    waveform, htk_compat=True, sample_frequency=sr, use_energy=False,
    window_type='hanning', num_mel_bins=mel_bins, dither=0.0,
    frame_shift=10)

  n_frames = fbank.shape[0]

  p = target_length - n_frames
  if p > 0:
    m = torch.nn.ZeroPad2d((0, 0, 0, p))
    fbank = m(fbank)
  elif p < 0:
    fbank = fbank[0:target_length, :]

  fbank = (fbank - (-4.2677393)) / (4.5689974 * 2)
  return fbank


def load_label(label_csv):
  with open(label_csv, 'r') as f:
    reader = csv.reader(f, delimiter=',')
    lines = list(reader)
  labels = []
  ids = []  # Each label has a unique id such as "/m/068hy"
  for i1 in range(1, len(lines)):
    id = lines[i1][1]
    label = lines[i1][2]
    ids.append(id)
    labels.append(label)
  return labels



def inference():
  class_name ='cough'
  top_k = 10
  # Assume each input spectrogram has 1024 time frames
  input_tdim = 1024
  checkpoint_path = 'pretrained_models/audio_mdl.pth'

  ast_mdl = ASTModel(label_dim=527, input_tdim=input_tdim, imagenet_pretrain=False, audioset_pretrain=False)

  print(f'Load checkpoint: {checkpoint_path}')

  checkpoint = torch.load(checkpoint_path, map_location='cuda')
  audio_model = torch.nn.DataParallel(ast_mdl, device_ids=[0])
  audio_model.load_state_dict(checkpoint)
  audio_model = audio_model.to(torch.device("cuda:0"))
  audio_model.eval()

  # Load the AudioSet label set
  label_csv = 'egs/audioset/data/class_labels_indices.csv'  # label and indices for audioset data
  labels = load_label(label_csv)

  files = glob.glob(f'/home/ubuntu/ast/sensi_datasets/env-classifier-benchmark-dataset/dataset_samples_by_classes/{class_name}/*.ogg')

  files = []
  base_path = f'/home/ubuntu/ast/sensi_datasets/env-classifier-benchmark-dataset/dataset_samples_by_classes/'
  classes_dirs  = os.listdir(base_path)
  for class_dir in classes_dirs:
    if class_dir == class_name:
      continue

    class_file_paths = os.path.join(base_path,class_dir)
    files.extend(glob.glob(class_file_paths + '/*.ogg'))

  labels_agg = []
  for i, audio_path in enumerate(files):

    feats = make_features(audio_path, mel_bins=128)  # shape(1024, 128)
    feats_data = feats.expand(1, input_tdim, 128)  # reshape the feature

    with torch.no_grad():
      with autocast():
        output = audio_model.forward(feats_data)
        output = torch.sigmoid(output)
    result_output = output.data.cpu().numpy()[0]
    sorted_indexes = np.argsort(result_output)[::-1]

    for k in range(top_k):
      # print('{}: {:.4f}'.format(np.array(labels)[sorted_indexes[k]], result_output[sorted_indexes[k]]))
      labels_agg.append(np.array(labels)[sorted_indexes[k]])
    # print(np.array(labels)[sorted_indexes][:5])

  counter = Counter(labels_agg)
  print(f"number of Slam (audio set label for object hit)  {counter['Slam']}/{len(files)}")
  print(f"number of Speech (audio set label for talk) {counter['Speech']}/{len(files)}")
  print(f"number of Television (audio set label for tv)  {counter['Television']}/{len(files)}")
  print(f"number of Throat clearing (audio set label for cough)  {counter['Throat clearing']}/{len(files)}")
  print(f"number of Cough (audio set label for cough)  {counter['Cough']}/{len(files)}")
  print(f"number of Snort (audio set label for snore)  {counter['Snort']}/{len(files)}")
  print(f"number of Shout (audio set label for non-verbal / verbal shout)  {counter['Shout']}/{len(files)}")
  print(f"number of Screaming (audio set label for non-verbal / verbal shout)  {counter['Screaming']}/{len(files)}")
  print(f"number of Sneeze (audio set label for cough)  {counter['Sneeze']}/{len(files)}")




  print(counter)

if __name__ == '__main__':
    inference()