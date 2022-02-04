import sys
import time
import wave
import pyaudio
import argparse
import numpy as np

import torch

import torchaudio

from dataset import valid_audio_transforms, greedy_decoder

class SpeechRecognizer:
    def __init__(self, model_file, kenlm_file):
        self.model = torch.jit.load(model_file)
        self.model.eval().to('cpu')

        self.hidden = (torch.zeros(1,1,1024), torch.zeros(1,1,1024))
        self.featurizer = get_featurizer(8000) 

    def annotate(self, wave_file):
        with torch.no_grad():
            waveform, _  = torchaudio.load(wave_file)
            mel_spec = valid_audio_transforms(waveform).unsqueeze(0) ##(1,1,feature,time)
            outputs, (ht, ct) = self.model(mel_spec, self.hidden)
            outputs = torch.nn.functional.softmax(outputs, dim=2) 
            text, _ = greedy_decoder(outputs)

            print(" ".join(text.split()))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="demoing the speech recognition engine in terminal.")
    parser.add_argument('--audio_file', type=str, default=None, required=True,
                        help='audio file to annotate')
    parser.add_argument('--model_file', type=str, default=None, required=True,
                        help='optimized file to load. use optimize_graph.py')
    parser.add_argument('--kenlm_file', type=str, default=None, required=False,
                        help='If you have an ngram lm use to decode')

    args = parser.parse_args()
    asr_engine = SpeechRecognizer(args.model_file, args.kenlm_file)
    asr_engine.annotate(args.audio_file)
