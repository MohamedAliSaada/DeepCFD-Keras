import numpy as np
np.random.seed(42)

from PIL import Image
import matplotlib.pyplot as plt
# load same img with small differnece (error)
img1 = Image.open("images.jpg").convert("RGB")
img2 = Image.open("images2.jpg").convert("RGB")

img1_arr=np.array(img1).astype(np.int16)
img2_arr=np.array(img2).astype(np.int16)

#get the error between two images 
error_arr = np.abs(img1_arr-img2_arr).astype(np.uint8)

error = Image.fromarray(error_arr)
# show the two images and error 

plt.subplot(1,3,1)
plt.imshow(img1)
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(img2)
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(error)
plt.axis("off")
#another example 

#my base image is all values 10
o_img =  np.ones((10,10)) *10

#make image that is 
p1 =  np.ones((10,3)) *7
p2 =  np.ones((10,3)) *5
p3 =  np.ones((10,4)) *10

m_img=np.concatenate([p1,p2,p3] , axis=1)

#plt the results and two images 

plt.subplot(1,3,1)
plt.imshow(o_img, cmap='jet' , vmin=0, vmax=10)
plt.colorbar()

plt.subplot(1,3,2)
plt.imshow(m_img, cmap='jet', vmin=0, vmax=10)
plt.colorbar()

error=(m_img == o_img ).astype(np.uint8)

plt.subplot(1,3,3)
plt.imshow(abs(m_img-o_img), cmap='gray')  #or use error
plt.colorbar()

plt.tight_layout()
plt.show()
