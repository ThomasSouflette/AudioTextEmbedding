from models.model import UnaterModel
import torch
from torch.utils.data import DataLoader
from torch.nn import functional as F
from torch.nn.utils import clip_grad_norm_

def main(opts):
    checkpoint = {}
    model = UnaterModel(opts.model_config, checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str,
                        help="path to model structure config json")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="path to model checkpoint (*.pt)")

    parser.add_argument('--config', required=True, help='JSON config files')
    
    args = parse_with_config(parser)
    main(args)