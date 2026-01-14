from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import numpy as np

class Terrain:
    # Class-level constants
    CLASS_NAMES = {
        0: 'grassy_terrain',
        1: 'marshy_terrain',
        2: 'rocky_terrain',
        3: 'sandy_terrain',
        4: 'urban'
    }

    def __init__(self, frame, model_path="type2/terrain_cls_int8.tflite"):
        self.org_img = Image.fromarray(frame) if not isinstance(frame, Image.Image) else frame
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.probs = None

    def preprocess(self, size=(128, 128)):
        img = self.org_img.resize(size)
        arr = np.array(img).astype(np.float32) / 255.0
        return np.expand_dims(arr, axis=0)

    def top_match(self):
        img = self.preprocess()
        self.interpreter.set_tensor(self.input_details[0]['index'], img)
        self.interpreter.invoke()
        self.probs = self.interpreter.get_tensor(self.output_details[0]['index'])[0]

        top_id = int(np.argmax(self.probs))
        return Terrain.CLASS_NAMES[top_id], round(self.probs[top_id], 2)

    def draw(self):
        if self.probs is None:
            raise ValueError("Run top_match() first to compute probabilities.")

        draw = ImageDraw.Draw(self.org_img)
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except OSError:
            font = ImageFont.load_default()

        y_offset = 10
        for i, name in Terrain.CLASS_NAMES.items():
            conf = self.probs[i]
            text = f"{name}: {conf:.2f}"
            text_w, text_h = draw.textsize(text, font=font)
            draw.rectangle([(8, y_offset-2), (8+text_w+4, y_offset+text_h)], fill="black")
            draw.text((10, y_offset), text, fill="white", font=font)
            y_offset += text_h + 5

        self.org_img.show()

    def ifsafe(self):
        name, prob = self.top_match()
        if name not in ['marshy_terrain', 'rocky_terrain'] and prob >= 0.5:  # use normalized prob
            print(f"Safe terrain: {name} with probability={prob}")
            return "safe"
        return "unsafe"