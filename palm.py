import os
from dotenv import load_dotenv, find_dotenv

import google.generativeai as palm

load_dotenv(find_dotenv())


api_key = os.environ["GOOGLE_API_KEY"]
palm.configure(api_key=api_key)


# choose a model and write details of available model.
models = [
    m for m in palm.list_models() if "generateText" in m.supported_generation_methods
]

print("there are {} models available".format(len(models)))
model = models[0].name
print(model)

# generate text
prompt = "Why sky is green?"
text = palm.generate_text(
    prompt=prompt,
    model=model,
    temperature=0.1,
    max_output_tokens=64,
    stop_sequences=["\n"],
)

print(text.result)
