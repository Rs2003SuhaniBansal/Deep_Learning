import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

image = np.random.rand(6,6)

print("Input image shape:", image.shape)

kernel = np.array([
    [-1, 0, 1],
    [-1, 0, 1],
    [-1, 0, 1]
])

print("\nKernel:")
print(kernel)

def convolution(image, kernel, stride=1):

    image_height, image_width = image.shape

    kernel_height, kernel_width = kernel.shape

    # Calculate output dimensions
    output_height = ((image_height - kernel_height) // stride) + 1

    output_width = ((image_width - kernel_width) // stride) + 1

    # output matrix
    output = np.zeros((output_height, output_width))

    # Slide kernel across image
    for i in range(0, output_height):

        for j in range(0, output_width):

            # Extract region
            region = image[
                i * stride : i * stride + kernel_height,
                j * stride : j * stride + kernel_width
            ]

            # Element-wise multiplication followed by summation
            output[i, j] = np.sum(region * kernel)

    return output

# CONV operation

feature_map = convolution(image,kernel)

print("\nOutput feature map shape:",
      feature_map.shape)

print("\nFeature map:")
print(feature_map)


plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)

plt.imshow(image, cmap="gray")

plt.title("Input Image (6 × 6)")
plt.axis("off")


plt.subplot(1, 2, 2)

plt.imshow(feature_map, cmap="gray")

plt.title("Feature Map")
plt.axis("off")

plt.show()


import numpy as np
import matplotlib.pyplot as plt

# MaxPool from scratch

def max_pool(image, pool_size=2, stride=2):

    image_height, image_width = image.shape

    # Calculate output dimensions
    output_height = ((image_height - pool_size) // stride) + 1

    output_width = ((image_width - pool_size) // stride) + 1

    # Create output matrix
    output = np.zeros((output_height, output_width))

    # Slide pooling window
    for i in range(output_height):

        for j in range(output_width):

            # Extract pooling region
            region = image[
                i * stride : i * stride + pool_size,
                j * stride : j * stride + pool_size
            ]

            # Take maximum value
            output[i, j] = np.max(region)

    return output

# MAX-POOL operation
pooled_feature_map = max_pool(feature_map,pool_size=2,stride=2)

print("\nAfter max pooling:")
print("Shape:", pooled_feature_map.shape)

print("\nPooled feature map:")
print(pooled_feature_map)

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)

plt.imshow(feature_map, cmap="gray")

plt.title("Feature Map")
plt.axis("off")

plt.subplot(1, 2, 2)

plt.imshow(pooled_feature_map,cmap="gray")

plt.title("MaxPool Output")
plt.axis("off")

plt.show()