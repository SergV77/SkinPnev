from settings import *
from function import *


def load_image(img_path, show=False):

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)                    # (height, width, channels)
    img_tensor = np.expand_dims(img_tensor, axis=0)         # (1, height, width, channels), add a dimension because the model expects this shape: (batch_size, height, width, channels)
    #img_tensor /= 255.                                      # imshow expects values in the range [0, 1]

    if show:
        plt.imshow(img_tensor[0])
        plt.axis('off')
        plt.show()

    return img_tensor


if __name__ == "__main__":
    # load model
    #model = load_model("db/models/model_13072021_2.hdf5")
    model = load_model("db/models/model_13072021_2.h5")

    # image path
    #img_path = 'db/test/pneumonia/NORMAL/IM-0095-0001.jpeg'
    #img_path = 'db/test/pneumonia/PNEUMONIA/person56_virus_112.jpeg'


    # load a single image
    #new_image = load_image(img_path)

    imagePatches = glob('db/test/pneumonia/**/*.jpeg', recursive=True)
    count1 = 0
    count2 = 0

    # check prediction
    for img in imagePatches:
        new_image = load_image(img)
        pred = model.predict(new_image)
        pred = np.argmax(pred, axis=1)
        print(img)
        print(pred)
        if pred[0] == 0:
            count1 += 1
        else:
            count2 += 1
        print('*'*100)
    print(count1)
    print(count2)