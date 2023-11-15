from dotenv import load_dotenv
import replicate
import gradio as gr
import form_variables as options

# load the environment variables from the .env file
load_dotenv()

def generative(relation, age, interests, budget, n):
    interests_ = f"interests: {interests}," if interests else ""
    recipient = f"for my {relation}" if relation else ""
    
    output = replicate.run(
        "meta/llama-2-70b-chat:02e509c789964a7ea8736978a43525956ef40397be9033abf9fd2badfe68c9e3",
        input={
            "debug": False,
            "top_k": 50,
            "top_p": 1,
            "prompt": f"suggest gifts {recipient}, age {int(age)}, {interests_} budget is {budget} usd.",
            "temperature": 0.25,
            "system_prompt": f"act as a personal assistant with a primary task of generating considerate and meaningful Christmas gift suggestions. The assistant should provide a curated list of up to {int(n)} items. The goal is to offer diverse and appealing options for users to choose from.",
            "max_new_tokens": 500,
            "min_new_tokens": -1
        }
    )

    output = list(output)       
    return ''.join(output)

demo = gr.Interface(
    fn=generative, 
    title="Suggest a christmas present!",
    description="Provide details below to gain suggestions for a christmas present!",
    inputs=[
            gr.Dropdown(options.relationship, label="Relationship", info="Relationship to the recipient"),
            gr.Number(minimum=0, maximum=100, label="Age"),
            gr.Text(label="Recipient's interests"), 
            gr.Number(minimum=1, maximum=200000, value=20, label="Budget", info="USD"),
            gr.Number(minimum=1, maximum=10, value=5, label="Number of suggestions", info="max 10")
        ], 
    outputs="text"
)

demo.launch(share=True)