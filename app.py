import streamlit as st
from foodclassifier import *

def preprocess_image(image: Image.Image):
    image = image.convert("RGB")
    image = transform(image) 
    image = image.unsqueeze(0) 
    return image

@st.cache_resource
def load_cnn():
    cnn = CNN()
    cnn.load_state_dict(torch.load('model.pth', weights_only=True, map_location=torch.device("cpu")))
    return cnn

model = load_cnn()

st.title(':green[Food] Classifier')
st.markdown("""
        Available Foods:
        - Baked Potato
        - Burger
        - Crispy Chicken
        - Donut
        - Fries
        - Hot Dog
        - Pizza
        - Sandwich
        - Taco
        - Taquito
        """)

result_container = st.container(border=True, height=500)

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg", "webp"])


with result_container:
    with st.chat_message('ai'):
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            st.image(image, caption="Uploaded Image", use_container_width=True)

            img_tensor = preprocess_image(image)

            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).item()
                converted_pred = IDX_TO_FOOD[pred]
            
            if converted_pred == 'Fries':
                st.write(f"Hm. This looks like some {converted_pred}!")
            else:
                st.write(f"Hm. This looks like a {converted_pred}!")

        else:
            st.write("Hello there! The name's :green[Foodie].")


