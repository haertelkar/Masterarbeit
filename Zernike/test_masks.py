import torch
from matplotlib import pyplot as plt

image_size = 38  # Größe des Bildes
center = 38 // 2 - 0.5  # 18.5 (patch geht von 0 bis 18, also 19x19)
r = center
patch_size = 19  # Größe des Patches

grid_x, grid_y = torch.meshgrid(torch.arange(image_size), torch.arange(image_size), indexing='ij')
masks = []
mask_total = torch.zeros((image_size, image_size), dtype=torch.bool)
mask_total = (grid_x - center) ** 2 + (grid_y - center) ** 2 <= r ** 2
for x in range(2):
    for y in range(2):
        mask = torch.zeros((patch_size, patch_size), dtype=torch.bool)
        mask = mask_total[x*patch_size:(x+1)*patch_size, y*patch_size:(y+1)*patch_size].clone()#.flatten().clone()
        masks.append(mask)

# Plotting the masks
plt.figure(figsize=(10, 10))
for i, mask in enumerate(masks):
    plt.subplot(2, 2, i + 1)
    plt.imshow(mask.view(patch_size, patch_size).cpu(), cmap='gray', interpolation='nearest')
    plt.title(f'Mask {i + 1}')
    plt.axis('off')
plt.tight_layout()
plt.savefig('masks.png')
plt.close()