import os
import replicate
import gradio as gr
os.environ["REPLICATE_API_TOKEN"] = "r8_P2tgO2HXbnjhSEjr8ueIzq2igdnOVQn0sbxNn"

def generative(prompt):
    output = replicate.run(
        "stability-ai/stable-diffusion:27b93a2413e7f36cd83da926f3656280b2931564ff050bf9575f1fdf9bcd7478",
        input={"prompt": prompt}
    )
    
    return output[0]

demo = gr.Interface(
    fn=generative, 
    inputs=["text"], 
    outputs="image"
)
    
demo.launch(share=True
