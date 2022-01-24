from models.model import ATEModel
from transformers import BertTokenizer


if __name__ == '__main__':
	model = ATEModel()
	tokenizer = BertTokenizer.from_pretrained("bert-base-cased")

	text='couille rances de chien de talu'
	input_ids = tokenizer.tokenize(text)
	token_types_ids = input_ids['token_types_ids']
	
	audio_file_name = '/content/drive/MyDrive/AUDIOCAPT/test.wav'
	audio_input, sr = sf.read(audio_file_name)
	
	embeddings = model.forward(input_ids, token_types_ids,
								audio_input, sr)

	print(embeddings)