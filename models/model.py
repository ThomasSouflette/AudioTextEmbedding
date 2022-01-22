import torch
from torch import nn
from models.utils_models import UnaterPreTrainedModel


class UnaterTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size,
                                            config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings,
                                                config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size,
                                                  config.hidden_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model
        # variable name and be able to load any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, position_ids, token_type_ids=None):
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = (words_embeddings
                      + position_embeddings
                      + token_type_embeddings)
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class UnaterModel(UnaterPreTrainedModel):
    """ Modification for Joint Vision-Language Encoding
    """
    def __init__(self, config, img_dim):
        super().__init__(config)
        self.embeddings = UnaterPreTrainedModel(config)
