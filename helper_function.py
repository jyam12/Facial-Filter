import cv2
import numpy as np
import matplotlib.pyplot as plt


# show the image
def show_image(image, name="",mask=None):
    if mask is not None:
        masked_image=cv2.bitwise_and(image, image, mask=mask)
    else:
        masked_image=image
    plt.figure()
    plt.imshow(masked_image)
    plt.title(f'{name}')
    plt.axis('off')
    plt.show()

# Save the image
def save_image(image, name):
    cv2.imwrite(f"./result/{name}.jpg", image)

# Prints the dimensions (height and width) of an image
def print_detail_of_image(image, name=""):
    height, width, _=image.shape
    print(f"{name}:\n height:{height} width:{width}")

# Padding the image with white pixel to achieve a square shape with the max size
def pad_image(image, height, width, max_pixel):
    pad_top=(max_pixel-height)//2
    pad_bottom=max_pixel-height-pad_top
    pad_left=(max_pixel-width)//2
    pad_right=max_pixel-width-pad_left
    
    padded_image=cv2.copyMakeBorder(image,pad_top, pad_bottom, pad_left, pad_right,
                                        cv2.BORDER_CONSTANT,value=(255, 255, 255))
    
    return padded_image

# Pads two images to have the same square dimensions, based on the maximum dimension of either image.
# def extend_shape(image1, image2):
#     height1, width1, _ = image1.shape
#     height2, width2, _ = image2.shape
#     max_pixel = max(height1, width1, height2, width2)

#     padded_image1 = pad_image(image1, height1, width1, max_pixel)
#     padded_image2 = pad_image(image2, height2, width2, max_pixel)
    
#     return padded_image1, padded_image2

# Merge the face_only & without_face to reconstruct the original image
# def merge_face(face_only,without_face):
#     lower_green = np.array([0, 200, 0])
#     upper_green = np.array([50, 255, 50])
#     mask = cv2.inRange(without_face, lower_green, upper_green)

#     mask_3ch = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)

#     # Extract face region from face_only using the mask
#     face_region = cv2.bitwise_and(face_only, mask_3ch)

#     # Extract non-face region from without_face
#     inverse_mask = cv2.bitwise_not(mask_3ch)
#     non_face_region = cv2.bitwise_and(without_face, inverse_mask)
    
#     # Combine the two regions
#     result_image= cv2.add(face_region, non_face_region)

#     return result_image

# Extract face region from face_only
def face_region_masks(image):
    lower_green=np.array([0, 200, 0])
    upper_green=np.array([50, 255, 50])
    mask=cv2.inRange(image, lower_green, upper_green)

    inverse_mask=cv2.bitwise_not(mask)

    return inverse_mask

# Comparing the oringial and result
def compare_result(oringial_image,result_image, name=""):
    plt.figure(figsize=(12, 8))
    plt.subplot(1,2, 1)
    plt.imshow(oringial_image)
    plt.title('Oringial')
    plt.subplot(1,2, 2)
    plt.imshow(result_image)
    plt.title(f'{name}')
    plt.axis('off')
    plt.show()
