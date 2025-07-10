import gradio as gr
from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
import matplotlib.pyplot as plt
import networkx as nx

# --------- 1. Sentence Classification ---------
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
labels = ["Air Pollution", "Water Conservation", "Climate Change", "Waste Management", "Wildlife Protection"]

def classify_sentence(sentence):
    result = classifier(sentence, candidate_labels=labels)
    return result['labels'][0]

# --------- 2. Image Generation ---------
image_gen = pipeline("text-to-image-generation", model="CompVis/stable-diffusion-v1-4")

def generate_image(prompt):
    return image_gen(prompt)[0]

# --------- 3. Named Entity Recognition + Graph Map ---------
ner_model = pipeline("ner", grouped_entities=True)

def ner_graph(text):
    entities = ner_model(text)
    G = nx.Graph()
    for ent in entities:
        G.add_node(ent['word'], label=ent['entity_group'])

    # Create dummy connections
    for i in range(len(entities) - 1):
        G.add_edge(entities[i]['word'], entities[i+1]['word'])

    # Plot graph
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', font_weight='bold')
    return plt

# --------- 4. Fill in the Blank (Masked Language Modeling) ---------
mask_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
mask_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
fill_mask = pipeline("fill-mask", model=mask_model, tokenizer=mask_tokenizer)

def fill_blank(text):
    if "[MASK]" not in text:
        return "Please include [MASK] in your sentence."
    result = fill_mask(text)
    return result[0]['sequence']

# --------- Gradio Interface ---------
with gr.Blocks() as demo:
    gr.Markdown("🌍 **Environmental AI Assistant - Powered by Hugging Face 🤗 + Gradio**")

    with gr.Tab("1️⃣ Sentence Classification"):
        sentence = gr.Textbox(placeholder="Enter a sentence related to environment")
        prediction = gr.Textbox(label="Prediction")
        classify_btn = gr.Button("Classify")
        classify_btn.click(classify_sentence, sentence, prediction)

    with gr.Tab("2️⃣ Image Generation"):
        prompt = gr.Textbox(placeholder="Describe the environmental scene")
        output_img = gr.Image()
        img_btn = gr.Button("Generate")
        img_btn.click(generate_image, prompt, output_img)

    with gr.Tab("3️⃣ NER with Graph Map"):
        ner_input = gr.Textbox(placeholder="Enter text for entity recognition")
        ner_output = gr.Plot()
        ner_btn = gr.Button("Extract Entities and Show Graph")
        ner_btn.click(ner_graph, ner_input, ner_output)

    with gr.Tab("4️⃣ Fill the Blank (Masked Language Modeling)"):
        mask_input = gr.Textbox(placeholder="Enter a sentence using [MASK] for missing word")
        mask_output = gr.Textbox(label="Predicted Sentence")
        mask_btn = gr.Button("Fill Blank")
        mask_btn.click(fill_blank, mask_input, mask_output)

demo.launch()
