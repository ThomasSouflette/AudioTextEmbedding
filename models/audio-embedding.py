import openl3
import soundfile as sf



if __name__ == '__main__':
	file_name = '/content/drive/MyDrive/AUDIOCAPT/test.wav'
	audio, sr = sf.read(file_name)
	embedding, timestamps = openl3.get_audio_embedding(audio, sr)
	print(embedding)