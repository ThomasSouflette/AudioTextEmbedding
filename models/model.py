import torch
from torch import nn

from transformers import AutoModel, BertLayer
import openl3

class MultiModalTransformer(nn.Module):
	def __init__(self, config):
		super().__init__()
		self.config = config

		layer = BertLayer(config)
		num_hidden_layer = 10
		self.layer = nn.ModuleList([copy.deepcopy(layer)
									for _ in range(num_hidden_layer)])

	def forward(self, input_, attention_mask,
				output_all_encoded_layers=True):
		all_encoder_layers = []
		hidden_states = input_
		for layer_module in self.layer:
			hidden_states = layer_module(hidden_states, attention_mask)
			if output_all_encoded_layers:
				all_encoder_layers.append(hidden_states)
		if not output_all_encoded_layers:
			all_encoder_layers.append(hidden_states)
		return all_encoder_layers

class ATEModel(nn.Module):
	
	def __init__(self, config):
		super().__init__()
		self.config = config

		self.textEmbedder = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
		self.encoder = MultiModalTransformer(config)

	def forward(self, input_ids,
				audio_input, sr):
		text_embedding  = self.textEmbedder(**input_ids)
		#norm_text_embedding = nn.LayerNorm(text_embedding.size())

		audio_embedding, timestamps = openl3.get_audio_embedding(audio_input, sr, embedding_size=512, content_type="env", hop_size=0.5)
		#norm_audio_embedding = nn.LayerNorm(audio_embedding.size())
		#return norm_audio_embedding, norm_text_embedding
		return audio_embedding, text_embedding


