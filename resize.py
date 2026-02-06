from PIL import Image
# source = 'imgs/lc001.png'
# # Load the image
# img = Image.open(source)

# # Resize the image
# resized_img = img.resize((340, 480), Image.ANTIALIAS)

# # Save the resized image
# resized_img.save("imgs/resized_lc001.png")

dst = 'imgs/LC001.jpg'
# Load the image
img = Image.open(dst)

# Resize the image
resized_img = img.resize((875, 584), Image.ANTIALIAS)

# Save the resized image
resized_img.save("imgs/resized_LC001.jpg")