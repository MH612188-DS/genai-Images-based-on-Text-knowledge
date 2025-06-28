GandharaGen: Avatar Generation using Knowledge-Based Stable Diffusion
GandharaGen is an AI-powered application that generates realistic historical avatars and backstories based on ancient Gandharan or Taxilan texts. It combines Large Language Models (LLMs), sentence embedding-based retrieval, and diffusion-based image generation to bring culturally rich characters to life.

This project generates photorealistic avatars based on historical textual knowledge using Stable Diffusion. It leverages the SG161222/Realistic_Vision_V5.1_noVAE model to create culturally accurate avatars guided by descriptions derived from ancient Gandhara heritage texts.

• Key Features
    • Context-Aware Prompt Enrichment using historical texts
    
    • Semantic Search with FAISS and SentenceTransformers
    
    • Realistic Image Generation using Stable Diffusion
    
    • Culturally Aligned Backstory Generation via Gemini Pro
    
    • Tkinter GUI for an intuitive and interactive experience

•	Background
The project blends generative AI with historical context retrieval. We use a knowledge base of historical texts to provide semantically rich prompts that guide avatar generation. This is particularly relevant for the cultural revival and visualization of figures from Gandhara art and history.

•	Tech Stack
Model: SG161222/Realistic_Vision_V5.1_noVAE

•	Diffusion Backend
HuggingFace diffusers library

•	Language Base
 Text prompts derived from Gandhara-era Buddhist/Hellenistic texts

•	UI: Gradio (optional)
  Platform: Google Colab (supports CPU/GPU)

•	-Workflow & Pipeline

Pipeline
1.	Input:
User inputs a character name or keyword (e.g., “Bodhisattva warrior”).

2.	Knowledge Retrieval:
Textual description is fetched from a curated historical dataset or prompt-engineered manually.

3.	Prompt Construction:
Text is used to create detailed prompts (e.g., “a photorealistic Gandharan monk, wearing traditional robes, 3D lighting, cinematic pose”).

4.	Avatar Generation:
Stable Diffusion generates images based on the prompt.

5.	Output:
Final avatar(s) are displayed and downloadable.

Workflow Overview
1. Text Upload & Chunking
The user uploads historical Gandharan text files (*.txt).

Text is split into manageable semantic chunks (typically ~100 words each) using sentence-based chunking.

2. Embedding & Indexing (Feature Extraction)
Each chunk is embedded into a high-dimensional semantic space using SentenceTransformer('all-MiniLM-L6-v2').
These embeddings are indexed using FAISS (Facebook AI Similarity Search) for fast and efficient vector retrieval.

3. Semantic Retrieval of Context
When the user enters a prompt (e.g. “a warrior monk from the Kushan period”), it is also embedded and compared with the indexed chunks.
The top-k (e.g., 3) most relevant chunks are retrieved as historical context.

4. Prompt Enrichment via Gemini
Gemini Pro (Gemini 1.5 Flash) is used to enrich the user's short prompt with cultural references, using the retrieved historical context.
This transforms a basic input into a detailed, culturally grounded prompt.

5. Backstory Generation
A second prompt is sent to Gemini to generate a short narrative backstory (5–7 sentences) for the character, consistent with the enriched prompt and retrieved context.

6. Image Generation with Stable Diffusion
The enriched prompt is passed to a fine-tuned Stable Diffusion pipeline (Realistic_Vision_V5.1) to generate a realistic avatar.

The model is loaded on GPU using PyTorch and diffusers.

🧪 Example Output
Prompt:

A Gandharan merchant

Generated Image:
🖼️ A bearded trader in Indo-Greek robes with embroidered sashes, holding scrolls and incense near a marketplace.

Backstory:

Born to a family of traders in ancient Sirkap, he dealt in lapis lazuli and silk between Taxila and Bactria. Fluent in Greek, Prakrit, and Aramaic, he was known for mediating temple disputes and financing Buddhist monastic art...

🖥️ GUI Features:

📁 Upload Gandharan text files

🖊️ Enter your character prompt

⚙️ Generate enriched prompts, backstories, and images

📸 View and save your generated avatar
