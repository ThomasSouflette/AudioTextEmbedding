from model.model import UnaterTextEmbeddings

def main(opts):
    checkpoint = {}
    model = UnaterTextEmbeddings(opts.model_config, checkpoint)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str,
                        help="path to model structure config json")
    parser.add_argument("--checkpoint", default=None, type=str,
                        help="path to model checkpoint (*.pt)")

    parser.add_argument('--config', required=True, help='JSON config files')
    
    args = parse_with_config(parser)
    main(args)