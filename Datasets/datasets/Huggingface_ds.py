from datasets import load_dataset



# NotMNIST
notmnist = load_dataset("notmnist")

# Omniglot
omniglot = load_dataset("omniglot")

# Google QuickDraw
quickdraw = load_dataset("quickdraw", "full")  # or a subset like "apple", "cat", etc.

# STL-10
stl10 = load_dataset("stl10")

# CelebA
celeba = load_dataset("celeb_a")

# LFW (Labeled Faces in the Wild)
lfw = load_dataset("lfw")

# Oxford Flowers 102
flowers = load_dataset("oxford_flowers102")
