


# --------------------- Img to text model -----------------------

# from transformers import pipeline
#
#
# def img2text(image_path):
#     # Use the 'from_pretrained' method to manually download the model to a specified directory
#     model = "Salesforce/blip-image-captioning-base"
#     img_to_text = pipeline("image-to-text", model=model)  # Custom cache directory
#
#     # Process the image and generate text
#     text = img_to_text(image_path)[0]["generated_text"]
#     print(text)
#     return text
#
#
# # Provide the correct image path
# img2text(r"E:\jatin documents\passportsize.jpg")  # Update with a full path if needed
