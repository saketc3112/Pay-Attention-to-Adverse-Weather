from PIL import Image


background = Image.open("/home/saket/Dense/CGLF_OOD/research/object_detection/weights_original_image.png")
overlay = Image.open("/home/saket/Dense/CGLF_OOD/research/object_detection/weights_gated.png")


newsize = (1920, 1024)
overlay = overlay.resize(newsize)
# Shows the image in image viewer
overlay.save("/home/saket/Dense/CGLF_OOD/research/object_detection/weights_gated.png")

background = background.convert("RGBA")
overlay = overlay.convert("RGBA")

new_img = Image.blend(background, overlay, 0.5)
#new_img.show()
new_img.save("/home/saket/Dense/CGLF_OOD/research/object_detection/vis.png")
