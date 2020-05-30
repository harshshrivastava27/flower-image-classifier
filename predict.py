import numpy as np
import tensorflow as tf
from tensorflow_hub import KerasLayer

def process_image(np_image):
    tf_image = tf.convert_to_tensor(np_image)
    tf_image = tf.image.resize(tf_image, (224, 224))
    tf_image = tf.cast(tf_image, dtype=tf.float32)
    tf_image /= 255.0
    return tf_image.numpy()

def predict(img_path, model, categories, top_k):
    from PIL import Image
    image = Image.open(img_path)
    np_image = np.asarray(image)
    processed_img = process_image(np_image)
    processed_img = np.expand_dims(processed_img, axis=0)
    predictions = model.predict(processed_img)[0]
    k_indexes = np.argsort(predictions)[-top_k:][::-1]
    return {categories[idx + 1]: predictions[idx] for idx in k_indexes}

def main(args):
    import json
    img_path = args.img_path
    model_path = args.model_path
    top_k = args.top_k
    category_names = args.category_names
    with open(category_names) as file:    
        categories = json.load(file)
    categories = {int(key): categories[key] for key in categories}
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': KerasLayer})
    predictions = predict(img_path, model, categories, top_k)
    for class_name in predictions:
        print(f'{class_name}: {predictions[class_name]*100.0:.3f}%')
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser(description='Predicts the top flower names from an image along with their corresponding probabilities.')
    parser.add_argument('img_path', help='Path to a image to predict the flower name')
    parser.add_argument('model_path', help='Path to a HDF5 file which stores a Keras model to make the prediction')
    parser.add_argument('--top_k', help='Return the top K most likely classes', default=3, type=int)
    parser.add_argument('--category_names', help='Path to a JSON file mapping labels to flower names', default='./label_map.json')
    args = parser.parse_args()
    main(args)