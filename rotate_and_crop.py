import math
import numpy as np
import cv2

def rotate_and_crop(img_array, angle, save_rot_only=False):
    """
    Function takes in an image and rotates it by a certain angle, crops accordingly
    INPUTS:
        ::np.array:: img_array          #numpy array of image
        ::float:: Angle                 #angle of rotation in degrees
        ::boolean:: save_rot_only       #whether or not to save the only rotated image
    OUTPUT:
        ::np.array::                    #rotated and cropped image
    """

    #Angle in this function below is in radians!
    def largest_rotated_rect(w, h, angle):
        """
        Given a rectangle of size wxh that has been rotated by 'angle' (in
        radians), computes the width and height of the largest possible
        axis-aligned rectangle within the rotated rectangle.

        Original JS code by 'Andri' and Magnus Hoff from Stack Overflow

        Converted to Python by Aaron Snoswell
        """

        quadrant = int(math.floor(angle / (math.pi / 2))) & 3
        sign_alpha = angle if ((quadrant & 1) == 0) else math.pi - angle
        alpha = (sign_alpha % math.pi + math.pi) % math.pi

        bb_w = w * math.cos(alpha) + h * math.sin(alpha)
        bb_h = w * math.sin(alpha) + h * math.cos(alpha)

        gamma = math.atan2(bb_w, bb_w) if (w < h) else math.atan2(bb_w, bb_w)

        delta = math.pi - alpha - gamma

        length = h if (w < h) else w

        d = length * math.cos(alpha)
        a = d * math.sin(alpha) / math.sin(delta)

        y = a * math.cos(gamma)
        x = y * math.tan(gamma)

        return (
            bb_w - 2 * x,
            bb_h - 2 * y
        )

    def rotate_image(image, angle):
        """
        Rotates an OpenCV 2 / NumPy image about it's centre by the given angle
        (in degrees). The returned image will be large enough to hold the entire
        new image, with a black background
        """

        # Get the image size
        # No that's not an error - NumPy stores image matricies backwards
        image_size = (image.shape[1], image.shape[0])
        image_center = tuple(np.array(image_size) / 2)

        # Convert the OpenCV 3x2 rotation matrix to 3x3
        rot_mat = np.vstack(
            [cv2.getRotationMatrix2D(image_center, angle, 1.0), [0, 0, 1]]
        )

        rot_mat_notranslate = np.matrix(rot_mat[0:2, 0:2])

        # Shorthand for below calcs
        image_w2 = image_size[0] * 0.5
        image_h2 = image_size[1] * 0.5

        # Obtain the rotated coordinates of the image corners
        rotated_coords = [
            (np.array([-image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2,  image_h2]) * rot_mat_notranslate).A[0],
            (np.array([-image_w2, -image_h2]) * rot_mat_notranslate).A[0],
            (np.array([ image_w2, -image_h2]) * rot_mat_notranslate).A[0]
        ]

        # Find the size of the new image
        x_coords = [pt[0] for pt in rotated_coords]
        x_pos = [x for x in x_coords if x > 0]
        x_neg = [x for x in x_coords if x < 0]

        y_coords = [pt[1] for pt in rotated_coords]
        y_pos = [y for y in y_coords if y > 0]
        y_neg = [y for y in y_coords if y < 0]

        right_bound = max(x_pos)
        left_bound = min(x_neg)
        top_bound = max(y_pos)
        bot_bound = min(y_neg)

        new_w = int(abs(right_bound - left_bound))
        new_h = int(abs(top_bound - bot_bound))

        # We require a translation matrix to keep the image centred
        trans_mat = np.matrix([
            [1, 0, int(new_w * 0.5 - image_w2)],
            [0, 1, int(new_h * 0.5 - image_h2)],
            [0, 0, 1]
        ])

        # Compute the tranform for the combined rotation and translation
        affine_mat = (np.matrix(trans_mat) * np.matrix(rot_mat))[0:2, :]

        # Apply the transform
        result = cv2.warpAffine(
            image,
            affine_mat,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR
        )

        return result

    def rotate(img_array,angle,save_rot_only=False):
        height, width = img_array.shape[0:2] #height and width of original image

        rotated_img = rotate_image(img_array,angle) #rotated image
        #print(largest_rotated_rect(width, height, angle*math.pi/180))

        rot_height, rot_width = rotated_img.shape[0:2] #height and width of rotated image
        cx = int(rot_width/2) #centre x-coordinate of rotated image
        cy = int(rot_height/2) #centre y-coordinate of rotated image
        wr,hr = largest_rotated_rect(width, height, angle*math.pi/180) #width and height of rectangle with largest area that is in rotated image (within the borders of the original image)

        begin_width = int(cx-0.5*wr) #left
        end_width = int(cx+0.5*wr) #right
        begin_height = int(cy-0.5*hr) #top
        end_height = int(cy+0.5*hr) #bottom

        img_cropped=rotated_img[begin_height:end_height, begin_width:end_width] #crop the image to rectangle with largest area

        if save_rot_only==True: #if you want to save the only rotated image
            #draw a rectangle to show the border of the rectangle with largest area that is inside the borders of original image
            start_point=(begin_width,begin_height)
            end_point=(end_width,end_height)
            # Blue color in BGR
            color = (255, 0, 0)
            # Line thickness of 2 px
            thickness = 2
            # Using cv2.rectangle() method
            # Draw a rectangle with blue line borders of thickness of 2 px
            rotated_img = cv2.rectangle(rotated_img, start_point, end_point, color, thickness)
            cv2.imwrite('only_rot.jpg', rotated_img) #save image

        return img_cropped

    rotated_cropped = rotate(img_array,angle)
    return rotated_cropped

"""
#How to use this code
img_array = cv2.imread('2_kitchen.jpg')
w = img_array.shape[1]
h = img_array.shape[0]
print("Width, height = ", w, h)
angle = 20
result=rotate_and_crop(img_array,angle)
cv2.imwrite('rotated.jpg', result)
"""
