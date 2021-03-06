from models.model import ATEModel
from transformers import AutoTokenizer, AutoModel, AutoConfig


import soundfile as sf


if __name__ == '__main__':
	config = AutoConfig.from_pretrained("bert-base-cased")
	model = ATEModel(config)
	tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")


	text='couille rances de chien de talu'
	#tockenized_sequence = tokenizer.tokenize(text, return_tensors='pt')
	inputs = tokenizer(text, return_tensors='pt')
	#encoded_inputs = inputs['input_ids']
	
	audio_file_name = '/content/drive/MyDrive/AUDIOCAPT/test.wav'
	audio_input, sr = sf.read(audio_file_name)
	
	audio_embedding, text_embedding = model.forward(inputs, audio_input, sr)

	print("audio_embedding:", audio_embedding, "\n\n", text_embedding)