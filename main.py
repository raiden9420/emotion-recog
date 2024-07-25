import os
from model.cnn_model import create_image_model
from model.rnn_model import create_audio_model
from model.multimodal_model import create_multimodal_model
from tensorflow.keras.models import save_model

# Paths to save models
image_model_path = 'models/image_model.h5'
audio_model_path = 'models/audio_model.h5'
multimodal_model_path = 'models/multimodal_model.h5'

# Create and save models
image_model = create_image_model()
audio_model = create_audio_model()
multimodal_model = create_multimodal_model()

# Save models
save_model(image_model, image_model_path)
save_model(audio_model, audio_model_path)
save_model(multimodal_model, multimodal_model_path)
