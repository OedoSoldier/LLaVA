import os

from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

model_path = "checkpoints/llava-vicuna-7b-v1.5-finetune"
model_path = os.path.expanduser(model_path)
model_name = get_model_name_from_path(model_path)
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path, None, model_name
)
print(model)
