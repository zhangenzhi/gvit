import openslide

img_path = 'dataset/paip/154000-2019-05-00-01-01.svs'
slides = openslide.OpenSlide(img_path)
print(slides.level_dimensions)