import matplotlib.pyplot as plt
from scipy.misc import imsave
import numpy as np

def generate_and_save_images(image, name, path):
  fig = plt.figure(figsize=(8,8))

  for i in range(image.shape[0]):
      plt.subplot(8, 8, i+1)
      plt.imshow(image[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
      plt.axis('off')
        
  plt.savefig('{}/{}.png'.format(path, name))
  # plt.show()
  
  
def inverse_transform(images):
  return (images+1.)/2.

def merge(images, size):
  h, w = images.shape[1], images.shape[2]
  if (images.shape[3] in (3,4)):
    c = images.shape[3]
    img = np.zeros((h * size[0], w * size[1], c))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w, :] = image
    return img
  elif images.shape[3]==1:
    img = np.zeros((h * size[0], w * size[1]))
    for idx, image in enumerate(images):
      i = idx % size[1]
      j = idx // size[1]
      img[j * h:j * h + h, i * w:i * w + w] = image[:,:,0]
    return img
  else:
    raise ValueError('in merge(images,size) images parameter '
                     'must have dimensions: HxW or HxWx3 or HxWx4')

def saveImages(images, size, path):
    images = inverse_transform(images)
    images = np.squeeze(merge(images, size))
    return imsave(path, images)

    # return imsave(inverse_transform(images), size, image_path)
    # image = np.squeeze(merge(images, size))
    # return scipy.misc.imsave(path, image)
