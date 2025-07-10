import streamlit as st
from transformers import pipeline, AutoModelForMaskedLM, AutoTokenizer
import networkx as nx
import matplotlib.pyplot as plt

# ---------- Sentence Classification ----------
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", framework="pt")
labels = ["Air Pollution", "Water Conservation", "Climate Change", "Waste Management", "Wildlife Protection"]

def classify(text):
    result = classifier(text, labels)
    return result['labels'][0]

# ---------- Image Generation Placeholder ----------
def generate_image(prompt):
    st.info("Image generation is disabled in Streamlit Cloud. Please use Hugging Face Spaces for this feature.")
    return None

# ---------- NER + Graph ----------
ner_pipeline = pipeline("ner", grouped_entities=True)

def draw_graph(text):
    ents = ner_pipeline(text)
    G = nx.Graph()
    for e in ents:
        G.add_node(e['word'])
    for i in range(len(ents)-1):
        G.add_edge(ents[i]['word'], ents[i+1]['word'])

    fig, ax = plt.subplots()
    nx.draw(G, with_labels=True, node_color='lightgreen', font_size=10, ax=ax)
    return fig

# ---------- Masked Language Model ----------
mask_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
mask_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
fill_mask = pipeline("fill-mask", model=mask_model, tokenizer=mask_tokenizer)

def fill_in_blank(text):
    if "[MASK]" not in text:
        return "Add [MASK] token to predict missing word."
    prediction = fill_mask(text)
    return prediction[0]['sequence']

# ---------- Streamlit UI ----------
st.set_page_config(page_title="Environmental AI Assistant")
st.title("🌱 Environmental AI Assistant")

tab1, tab2, tab3, tab4 = st.tabs([
    "1️⃣ Sentence Classification", 
    "2️⃣ Image Generation", 
    "3️⃣ NER with Graph", 
    "4️⃣ Fill in the Blank"
])

with tab1:
    st.subheader("Classify Environmental Sentence")
    inp = st.text_input("Enter a sentence")
    if st.button("Classify"):
        st.success(f"Prediction: **{classify(inp)}**")

with tab2:
    st.subheader("Image Generation (Not available in Streamlit Cloud)")
    prompt = st.text_input("Enter image prompt")
    if st.button("Generate Image"):
        generate_image(prompt)

with tab3:
    st.subheader("Named Entity Recognition Graph")
    ner_input = st.text_area("Enter text")
    if st.button("Visualize Graph"):
        fig = draw_graph(ner_input)
        st.pyplot(fig)

with tab4:
    st.subheader("Fill in the Blank (Masked Language)")
    masked = st.text_input("Enter a sentence with [MASK]")
    if st.button("Fill Blank"):
        result = fill_in_blank(masked)
        st.info(result)
