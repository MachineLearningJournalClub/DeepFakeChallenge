import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import librosa
import librosa.display
from tqdm import tqdm, tqdm_notebook
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("main_path", help="Path with all the audio folders")
args = parser.parse_args()

main_path = args.main_path

paths = os.listdir(main_path)

def check_fake_by_audio(a,b,sr=20000):
  audio_array_a, sample_rate = librosa.load(main_path+'/'+path+'/'+a+'/audio.wav', sr=sr)
  audio_array_b, sample_rate = librosa.load(main_path+'/'+path+'/'+b+'/audio.wav', sr=sr)
  if audio_array_a.shape==audio_array_b.shape:
    if (audio_array_a==audio_array_b).all():
      return 0
    else:
      return 1.5 #distinguo quelli fatti bene da quelli fatti male
  else:
    return 1

for path in paths:
  df = pd.read_json(main_path+'/'+path+'/metadata.json')
  df = df.transpose().reset_index().sort_values('index').reset_index(drop=True)

  fake_by_audio = np.zeros(len(df))
  for i in range(len(df)):
    if df.loc[i,'label']=='FAKE':
      a = df.loc[i,'index'][:-4]
      b = df.loc[i,'original'][:-4]
      try:
        fake_by_audio[i]=check_fake_by_audio(a,b)
      except:
        with open(main_path+'/no_label.txt','a') as file:
          file.write(path+', '+str(i)+', '+df.iloc[i]['index'][:-4]+'\n')
  
  df['fake_by_audio']=np.floor(fake_by_audio)
  df.to_json(main_path+path+'/metadata_plus.json')

  fk_p = df.groupby('label').count().loc['FAKE','index']/len(df)
  fk_a_p = df.groupby('fake_by_audio').count().loc[1.0,'index']/len(df)

  with open(main_path+'/audio_stats.txt','a') as file:
    file.write(path+','+str(fk_p)+','+str(fk_a_p)+','+str(len(df))+'\n')