# the flan t5 based fine tunned model "lmsys/fastchat-t5-3b-v1.0" having 3biliion parameters taken from huggingface and ran on my local machine
# model from April 2023

# source: https://www.youtube.com/watch?v=Ay5K4tog5NQ
# Checked and and ran on ipython 
""" A couple of libraries had to be installed: 
"""

import os
from huggingface_hub import hf_hub_download
HUGGING_FACE_API_KEY = os.environ.get("HUGGING_FACE_API_KEY")
model_id = "lmsys/fastchat-t5-3b-v1.0"
filenames = [
        "pytorch_model.bin", "added_tokens.json", "config.json", "generation_config.json", 
        "special_tokens_map.json", "spiece.model", "tokenizer_config.json"
]
# downloading the model and its files: the names of the files can be ssen on hugging face website: https://huggingface.co/lmsys/fastchat-t5-3b-v1.0/tree/main
# model card and all details are also there. 

for filename in filenames:
        downloaded_model_path = hf_hub_download(
                    repo_id=model_id,
                    filename=filename,
                    token=HUGGING_FACE_API_KEY
        )

        print(downloaded_model_path)

"""
Run the LLM
Now let's try running the model. But before we do that, let's disable the Wi-Fi.

from utils import check_connectivity, toggle_wifi
import time

print(check_connectivity())
toggle_wifi("off")
time.sleep(0.5)
print(check_connectivity())
"""
#tokenize and get the inference out of this
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained(model_id, legacy=False)
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

pipeline = pipeline("text2text-generation", model=model, device=-1, tokenizer=tokenizer, max_length=1000)

pipeline("What are competitors to Apache Kafka?")
pipeline("""My name is Mark.
I have brothers called David and John and my best friend is Michael.
Using only the context above. Do you know if I have a sister?    
""")