from models.model import ATEModel
from transformers import AutoTokenizer, AutoModel

import soundfile as sf


if __name__ == '__main__':
	model = ATEModel()
	tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")

	text='couille rances de chien de talu'
	#tockenized_sequence = tokenizer.tokenize(text, return_tensors='pt')
	inputs = tokenizer(text)
	encoded_inputs = inputs['inputs_id']
	
	audio_file_name = '/content/drive/MyDrive/AUDIOCAPT/test.wav'
	audio_input, sr = sf.read(audio_file_name)
	
	embeddings = model.forward(encoded_inputs, audio_input, sr)

	print(embeddings)