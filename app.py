import streamlit as st
import random
import requests
import io
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from peft import PeftModel, PeftConfig
import torch


model = None
processor = None

st.title("Image Captioner - Caption the images")
st.markdown("Link to the model - [Image-to-Caption-App on 🤗 Spaces](https://huggingface.co/spaces/Shrey23/Image-Captioning)")


class UI:
    def __init__(self):
        mod = Model()
        mod.load_model()

    def displayUI(self):
        image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])
        if image is not None:

            input_image = Image.open(image) #read image
            st.image(input_image) #display image

            with st.spinner("🤖 AI is at Work! "):
                device = "cuda" if torch.cuda.is_available() else "cpu"
                print(device, 1)
                inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)

                print(2)
                pixel_values = inputs.pixel_values

                print(3)
                generated_ids = model.generate(pixel_values=pixel_values, max_length=25)
                generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                st.write(generated_caption)
                
            st.success("Here you go!")
            st.balloons()
        else:
            st.write("Upload an Image")

        st.caption("Made with ❤️ by @1littlecoder. Credits to 🤗 Spaces for Hosting this ")
                        

class Model:
    def load_model(self):
        peft_model_id = "Shrey23/Image-Captioning"
        config = PeftConfig.from_pretrained(peft_model_id)
        global model
        global processor
        model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, low_cpu_mem_usage=True, torch_dtype=torch.float16) #, device_map="auto", load_in_8bit=True
        model = PeftModel.from_pretrained(model, peft_model_id)
        processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")

    def query(self , payload):
        response = requests.post(self.API_URL, headers=self.headers, json=payload)
        return response.content

    def generate_response(self, prompt):
        image_bytes = self.query({ "inputs": prompt, })
        return io.BytesIO(image_bytes)


def main():
    ui = UI()
    ui.displayUI()

if __name__ == "__main__":
    main()
