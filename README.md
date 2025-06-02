GandharaGen: Avatar Generation using Knowledge-Based Stable Diffusion
This project generates photorealistic avatars based on historical textual knowledge using Stable Diffusion. It leverages the SG161222/Realistic_Vision_V5.1_noVAE model to create culturally accurate avatars guided by descriptions derived from ancient Gandhara heritage texts.

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
