import matplotlib.pyplot as plt
import PIL
import os

example = "complex"
# load image
image = PIL.Image.open(f"paper_examples/{example}.jpeg")
# resize to 336x336
image = image.resize((336, 336))
# split into 16x16 patches
patches = [
    image.crop((x, y, x + 16, y + 16))
    for x in range(0, 336, 16)
    for y in range(0, 336, 16)
]
# save patches
for i, patch in enumerate(patches):
    if not os.path.exists(f"paper_examples/{example}"):
        os.makedirs(f"paper_examples/{example}")
    patch.save(f"paper_examples/{example}/patch_{i}.png")
