import streamlit as st  #Web App
from PIL import Image #Image Processing
import numpy as np #Image Processing 
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch
from peft import PeftModel, PeftConfig


#title
st.title("Image Captioner - Caption the images")

st.markdown("Link to the model - [Image-to-Caption-App on 🤗 Spaces](https://huggingface.co/spaces/Shrey23/Image-Captioning)")

#image uploader
image = st.file_uploader(label = "Upload your image here",type=['png','jpg','jpeg'])


@st.cache
def load_model():
    peft_model_id = "Shrey23/Image-Captioning"
    config = PeftConfig.from_pretrained(peft_model_id)
    model = Blip2ForConditionalGeneration.from_pretrained(config.base_model_name_or_path, device_map="auto") #, load_in_8bit=True
    model = PeftModel.from_pretrained(model, peft_model_id)

    processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
    return processor, model
    
processor, model = load_model() #load model

if image is not None:

    input_image = Image.open(image) #read image
    st.image(input_image) #display image

    with st.spinner("🤖 AI is at Work! "):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
        pixel_values = inputs.pixel_values


        generated_ids = model.generate(pixel_values=pixel_values, max_length=25)
        generated_caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        st.write(generated_caption)
        
    st.success("Here you go!")
    st.balloons()
else:
    st.write("Upload an Image")

st.caption("Made with ❤️ by @1littlecoder. Credits to 🤗 Spaces for Hosting this ")
