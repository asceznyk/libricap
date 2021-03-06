import argparse
from collections import OrderedDict

import torch

from hparams import *
from model import SpeechRecognizer

def main(args):
    print("loading model from", args.model_checkpoint)

    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu')) 
    model = SpeechRecognizer(**hparams)

    model_state_dict = checkpoint['model']
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k.replace("model.", "") # remove `model.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())

    print(f"{n_params} parameters")

    print("tracing model...") 

    traced_model = torch.jit.trace(model, torch.rand(1, 1, hparams['n_feats'], 300)) 
    
    print("saving to", args.save_path)
    
    traced_model.save(args.save_path)
    
    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="optimizing model graphs")
    parser.add_argument('--model_checkpoint', type=str, default=None, required=True,
                        help='Checkpoint of model to optimize')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='path to save optmized model')

    args = parser.parse_args()
    main(args)
