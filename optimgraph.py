import argparse

from collections import OrderedDict

import torch

from model import SpeechRecognizer

def main(args):
    print("loading model from", args.model_checkpoint)
    checkpoint = torch.load(args.model_checkpoint, map_location=torch.device('cpu')) 
    model = SpeechRecognizer(**SpeechRecognizer.hyper_params)

    model_state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()
    for k, v in model_state_dict.items():
        name = k.replace("model.", "") # remove `model.`
        new_state_dict[name] = v

    model.load_state_dict(new_state_dict)

    model.eval() 

    print("Tracing model...") 
    traced_model = torch.jit.trace(model, torch.rand(1, 1, 81, 300)) 
    print("Saving to", args.save_path)
    traced_model.save(args.save_path)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="optimizing model graphs")
    parser.add_argument('--model_checkpoint', type=str, default=None, required=True,
                        help='Checkpoint of model to optimize')
    parser.add_argument('--save_path', type=str, default=None, required=True,
                        help='path to save optmized model')

    args = parser.parse_args()
    main(args)
