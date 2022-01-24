import torch
from torch import nn

from transformers import BertModel
import openl3

class ATEModel(nn.Module):
	
	def __init__(self):
		super().__init__()
		self.textEmbedder = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states = True,
                                  )


	def forward(self, input_ids,
				audio_input, sr):
		text_embedding  = self.textEmbedder(input_ids)

		audio_embedding, timestamps = openl3.get_audio_embedding(audio_input, sr)
		return audio_embedding, text_embedding