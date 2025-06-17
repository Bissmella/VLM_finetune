from huggingface_hub import login
login("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")

from huggingface_hub import hf_hub_download

hf_hub_download(repo_id="LEVI-Project/sft-data", filename="sft-data.zip")