from transformers import BertTokenizer, BertModel
import pandas as pd
import numpy as np
import nltk
import torch




tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

texts = ["bank",
         "The river bank was flooded.",
         "The bank vault was robust.",
         "He had to bank on her for support.",
         "The bank was out of money.",
         "The bank teller was a man."]


def bert_text_preparation(text, tokenizer):

    marked_text = "[CLS] " + text + " [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
    segments_ids = [1]*len(indexed_tokens)

    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    return tokenized_text, tokens_tensor, segments_tensors
    



def get_bert_embeddings(tokens_tensor, segments_tensors, model):

    # Gradient calculation id disabled
    # Model is in inference mode
    with torch.no_grad():
        outputs = model(tokens_tensor, segments_tensors)
        # Removing the first hidden state
        # The first state is the input state
        hidden_states = outputs[2][1:]

    # Getting embeddings from the final BERT layer
    token_embeddings = hidden_states[-1]
    # Collapsing the tensor into 1-dimension
    token_embeddings = torch.squeeze(token_embeddings, dim=0)
    # Converting torchtensors to lists
    list_token_embeddings = [token_embed.tolist() for token_embed in token_embeddings]

    return list_token_embeddings





if __name__ == '__main__':
    target_word_embeddings = []
    """
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_config", type=str,
                        help="path to model structure config json")
    """

    for text in texts:
        tokenized_text, tokens_tensor, segments_tensors = bert_text_preparation(text, tokenizer)
        list_token_embeddings = get_bert_embeddings(tokens_tensor, segments_tensors, model)
    
        # Find the position 'bank' in list of tokens
        word_index = tokenized_text.index('bank')
        # Get the embedding for bank
        word_embedding = list_token_embeddings[word_index]

        target_word_embeddings.append(word_embedding)
        
    print(target_word_embeddings)