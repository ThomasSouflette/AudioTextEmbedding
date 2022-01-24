import torch
from torch import nn

from transformers import AutoModel
import openl3

class ATEModel(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.textEmbedder = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")


	def forward(self, input_ids,
				audio_input, sr):
		text_embedding  = self.textEmbedder(**input_ids)

		audio_embedding, timestamps = openl3.get_audio_embedding(audio_input, sr, embedding_size=512, content_type="env", hop_size=0.5)
		return audio_embedding, text_embedding