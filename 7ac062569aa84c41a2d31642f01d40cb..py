import streamlit as st
from PIL import Image
import requests
import io
import matplotlib.pyplot as plt
import networkx as nx

st.title("🌱 Environmental NLP & Image Generation Hub")

# ---- 1️⃣ Sentence Classification ----
st.header("1️⃣ Sentence Classification")

sentence = st.text_area("Enter a sentence related to environment for classification:")
if st.button("Classify Sentence"):
    # Placeholder (replace with Hugging Face API call)
    categories = ["Climate", "Pollution", "Energy", "Biodiversity", "Policy"]
    st.success(f"Predicted Category: {categories[0]}")

# ---- 2️⃣ Image Generation ----
st.header("2️⃣ Image Generation")

image_prompt = st.text_input("Enter prompt for environmental image generation:")

if st.button("Generate Image"):
    # Using a direct URL for placeholder without downloading
    image_url = "https://via.placeholder.com/400x300.png?text=Generated+Environmental+Image"

    # Display image directly from URL
    st.image(image_url, caption="Generated Image", use_column_width=True)

# ---- 3️⃣ Named Entity Recognition and Graph Map ----
st.header("3️⃣ Named Entity Recognition (NER) & Graph Map")

ner_sentence = st.text_area("Enter a sentence for NER:")
if st.button("Analyze NER & Show Graph"):
    # Placeholder NER result (replace with Hugging Face API)
    entities = [("Climate Change", "ENV_CONCEPT"), ("Paris Agreement", "POLICY")]

    st.write("**Detected Entities:**")
    for entity, label in entities:
        st.write(f"- {entity} ({label})")

    # Generate a simple graph map
    G = nx.Graph()
    for entity, label in entities:
        G.add_node(entity, label=label)
    G.add_edges_from([(entities[0][0], entities[1][0])])

    pos = nx.spring_layout(G)
    labels = nx.get_node_attributes(G, 'label')

    plt.figure(figsize=(5, 4))
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', node_size=2500, font_size=10)
    nx.draw_networkx_edge_labels(G, pos, edge_labels={(entities[0][0], entities[1][0]): "related"})
    st.pyplot(plt)

# ---- 4️⃣ Fill the Blank using Mask ----
st.header("4️⃣ Fill the Blank using Mask")

masked_sentence = st.text_area("Enter sentence with [MASK] related to environment:")
if st.button("Fill the Mask"):
    # Placeholder (replace with Hugging Face Mask Filling API)
    filled_sentence = masked_sentence.replace("[MASK]", "climate change")
    st.success(f"Filled Sentence: {filled_sentence}")

# Footer
st.markdown("---")
st.markdown("✅ Ready to deploy on **Hugging Face Spaces**.\nContact if you need inference API connection for live model calls.")
