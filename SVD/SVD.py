from PIL import Image
from matplotlib import pyplot as plt
from scipy.linalg import svd
import scipy as scipy
import numpy as np

img = Image.open('image.jpg')
im=img.convert('L')
print(img.format)
print(img.mode)
print(img.size)
U, s, VT = np.linalg.svd(im,full_matrices=False)
print(U)
print(s)
print(VT)
s = np.diag(s)
a = []
print(f"U shape: {U.shape}, s shape: {s.shape}, VT shape: {VT.shape}")
for k in [1,5,10,20,40] :
    a += [U[:,:k]@s[:k,:k]@VT[:k,:]]

figure=plt.figure(figsize=(10,7))
figure.set_facecolor('white')

figure.add_subplot(3,2,1)
plt.imshow(im,cmap='gray')
plt.title('Original Image')
plt.axis('off')

figure.add_subplot(3,2,2)
plt.imshow(a[0],cmap='gray')
plt.title('K=1')
plt.axis('off')

figure.add_subplot(3,2,3)
plt.imshow(a[1],cmap='gray')
plt.title('K=5')
plt.axis('off')

figure.add_subplot(3,2,4)
plt.imshow(a[2],cmap='gray')
plt.title('K=10')
plt.axis('off')

figure.add_subplot(3,2,5)
plt.imshow(a[3],cmap='gray')
plt.title('K=20')
plt.axis('off')

figure.add_subplot(3,2,6)
plt.imshow(a[4],cmap='gray')
plt.title('K=40')
plt.axis('off')

plt.show()
plt.subplots_adjust(left=0.1,
                    bottom=0.1, 
                    right=0.9, 
                    top=0.9, 
                    wspace=0.5, 
                    hspace=0.5)

# Write out the plots as an image
plt.savefig('plots.png')