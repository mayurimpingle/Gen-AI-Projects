from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import torch
import gradio as gr

model_name = "facebook/m2m100_418M"

model = M2M100ForConditionalGeneration.from_pretrained(model_name)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)

# Dictionary mapping language codes to full names
languages = {
    "en": "English",
    "it": "Italian",
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "hi": "Hindi",
    "gu": "Gujarati",
    "mr": "Marathi",
    "ru": "Russian",
    "zh": "Chinese"
}

def translate(text, src_lang, tgt_lang):
    # Convert full name back to code
    src_code = [k for k, v in languages.items() if v == src_lang][0]
    tgt_code = [k for k, v in languages.items() if v == tgt_lang][0]

    inputs = tokenizer(text, return_tensors="pt", padding=True)
    inputs["forced_bos_token_id"] = tokenizer.get_lang_id(tgt_code)
    with torch.no_grad():
        translated = model.generate(**inputs)
    translation = tokenizer.decode(translated[0], skip_special_tokens=True)
    return translation

# Gradio interface
interface = gr.Interface(
    fn=translate,
    inputs=[
        gr.Textbox(label="Text to Translate", lines=4),
        gr.Dropdown(label="Select Source Language", choices=list(languages.values()), value="English"),
        gr.Dropdown(label="Select Target Language", choices=list(languages.values()), value="French"),
    ],
    outputs=gr.Textbox(label="Translated Text"),
    title="M2M-100 Language Translator",
    description="Choose source and target languages, then enter text to get the translation."
)

interface.launch(debug=True, share=True)
