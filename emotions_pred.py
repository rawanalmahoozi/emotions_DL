import gradio as gr
import numpy as np
import keras

model = keras.models.load_model('emotions_model.keras')
class_names = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

def predict(img):
    img = img.convert('RGB').resize((48, 48))
    img = np.array(img) / 255
    img = img.reshape(1, 48, 48, 3)

    pred = model.predict(img)[0]

    result = {}
    for name, p in zip(class_names, pred):
        result[name] = float(p)

    return result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(sources=["webcam", "upload"], type="pil"),
    outputs=gr.Label(num_top_classes=7),
)

demo.launch(share=True)