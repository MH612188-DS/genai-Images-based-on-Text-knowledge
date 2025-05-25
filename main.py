import os
import re
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter.scrolledtext import ScrolledText
from PIL import Image, ImageTk
from dotenv import load_dotenv
import google.generativeai as genai
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import torch
from diffusers import StableDiffusionPipeline

# Load API Key
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-1.5-flash-latest"

genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel(MODEL_NAME)

def chunk_text(text, max_words=100):
    sentences = re.split(r'(?<=[.!?]) +', text)
    chunks, current_chunk, current_len = [], [], 0
    for sentence in sentences:
        word_count = len(sentence.split())
        if current_len + word_count > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_len = word_count
        else:
            current_chunk.append(sentence)
            current_len += word_count
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def load_embedder_and_index(chunks):
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embedder.encode(chunks, show_progress_bar=False)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    return embedder, index, embeddings

def retrieve_context(query, embedder, index, chunks, top_k=3):
    query_embedding = embedder.encode([query])
    D, I = index.search(query_embedding, top_k)
    return [chunks[i] for i in I[0]]

def generate_enriched_prompt(user_prompt, context):
    context_text = "\n".join(context)
    prompt = f"""
You are a historian helping enrich character prompts for image generation.

Context from Gandhara/Taxila history:
{context_text}

User Prompt:
{user_prompt}

Now write an enriched, vivid version of the prompt with cultural references:
"""
    return model.generate_content(prompt).text.strip()

def generate_backstory(enriched_prompt):
    backstory_prompt = f"""
Write a culturally appropriate backstory (5–7 sentences) for the following character:

{enriched_prompt}
"""
    return model.generate_content(backstory_prompt).text.strip()

def generate_image(prompt, pipe, output_path="avatar.png"):
    image = pipe(prompt).images[0]
    image.save(output_path)
    return output_path

def load_image_pipeline():
    if not torch.cuda.is_available():
        raise SystemError("CUDA is not available. Please check your GPU or driver installation.")

    pipe = StableDiffusionPipeline.from_pretrained(
        "SG161222/Realistic_Vision_V5.1_noVAE",
        torch_dtype=torch.float16,
        safety_checker=None
    ).to("cuda")  # Move model to GPU

    pipe.enable_xformers_memory_efficient_attention()


    return pipe


class GandharaGenApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GandharaGen - Historical Avatar Creator")

        self.text_data = ""
        self.chunks = []
        self.embedder = None
        self.index = None
        self.pipe = load_image_pipeline()

        tk.Button(root, text="📁 Upload Gandhara Texts", command=self.upload_files).pack(pady=5)
        self.prompt_entry = tk.Entry(root, width=60)
        self.prompt_entry.insert(0, "Describe your Gandharan character...")
        self.prompt_entry.pack(pady=5)

        tk.Button(root, text="⚙️ Generate Avatar", command=self.process_prompt).pack(pady=5)

        self.output_text = ScrolledText(root, height=10, wrap=tk.WORD)
        self.output_text.pack(padx=10, pady=5, fill=tk.BOTH)

        self.image_label = tk.Label(root)
        self.image_label.pack(pady=10)

    def upload_files(self):
        filepaths = filedialog.askopenfilenames(filetypes=[("Text files", "*.txt")])
        combined_text = ""
        for file in filepaths:
            with open(file, "r", encoding="utf-8") as f:
                combined_text += "\n" + f.read()
        self.text_data = combined_text
        self.chunks = chunk_text(self.text_data)
        self.embedder, self.index, _ = load_embedder_and_index(self.chunks)
        messagebox.showinfo("Upload Successful", f"Loaded {len(self.chunks)} text chunks.")

    def process_prompt(self):
        user_prompt = self.prompt_entry.get().strip()
        if not self.chunks:
            messagebox.showwarning("No Context", "Please upload text files first.")
            return
        if not user_prompt:
            messagebox.showwarning("Empty Prompt", "Please describe your Gandharan character.")
            return

        self.output_text.delete("1.0", tk.END)

        context = retrieve_context(user_prompt, self.embedder, self.index, self.chunks)
        enriched_prompt = generate_enriched_prompt(user_prompt, context)
        backstory = generate_backstory(enriched_prompt)

        self.output_text.insert(tk.END, "🎯 Enriched Prompt:\n" + enriched_prompt + "\n\n")
        self.output_text.insert(tk.END, "📜 Backstory:\n" + backstory + "\n")

        avatar_path = generate_image(enriched_prompt, self.pipe)
        image = Image.open(avatar_path)
        image = image.resize((256, 256))
        self.tk_image = ImageTk.PhotoImage(image)
        self.image_label.config(image=self.tk_image)

# Run the app
if __name__ == "__main__":
    root = tk.Tk()
    app = GandharaGenApp(root)
    root.mainloop()
