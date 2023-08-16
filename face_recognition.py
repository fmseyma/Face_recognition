import sys

from unicodedata import name
from pymongo import MongoClient
from bson import json_util
import json
import face_recognition
import cv2
import os
import numpy as np
import shutil
import math
from sklearn import neighbors
import pickle
from PIL import Image, ImageDraw
from face_recognition.face_recognition_cli import image_files_in_folder

names = []
encodes = []
n = 0
args: list[int] = sys.argv


def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            unknown_face_encodings = face_recognition.face_encodings(img)[0]
            splitNames = filename.split(".")
            item_1 = {"name": str(splitNames[0]), "Face_map": unknown_face_encodings.tolist()}
            collection.insert_one(item_1)
            shutil.move('../face_recognition/to-be-read/' + filename,
                        "../face_recognition//readed/" + filename)
            # os.remove(filename)
    # connection.close()


def read_from_dv():
    cursor = collection.find({})
    for document in cursor:
        names.append(document["name"])
        encodes.append(document["Face_map"])
    print(names)
    print(encodes)


ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'JPG'}


def train(train_dir, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.
    :param train_dir: directory that contains a sub-directory for each known person, with its name.
     (View in source code to see train_dir example tree structure)
     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...
    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """
    X = []
    y = []
    print("x=[] ve y=[]")
    # Loop through each person in the training set
    for class_dir in os.listdir(train_dir):
        print("for class_dir in os.listdir(train_dir):")
        if not os.path.isdir(os.path.join(train_dir, class_dir)):
            print("if not os.path.isdir(os.path.join(train_dir, class_dir)):")
            continue

        # Loop through each training image for the current person
        for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):
            print("for img_path in image_files_in_folder(os.path.join(train_dir, class_dir)):")
            image = face_recognition.load_image_file(img_path)
            print("image")
            face_bounding_boxes = face_recognition.face_locations(image)
            print("face_bounding_boxes")
            if len(face_bounding_boxes) != 1:
                print("if len(face_bounding_boxes) != 1:")
                # If there are no people (or too many people) in a training image, skip the image.
                if verbose:
                    print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(
                        face_bounding_boxes) < 1 else "Found more than one face"))
            else:
                print("else")
                # Add face encoding for current image to the training set
                X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
                y.append(class_dir)

    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    print("knn_clf")
    knn_clf.fit(X, y)
    print("knn_clf.fit")

    # Save the trained KNN classifier
    if model_save_path is not None:
        print("if model_save_path is not None:")
        with open(model_save_path, 'wb') as f:
            print("with open(model_save_path, 'wb') as f:")
            pickle.dump(knn_clf, f)
            print("pickle.dump(knn_clf, f)")

    return knn_clf


def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.5):
    """
    Recognizes faces in given image using a trained KNN classifier
    :param X_frame: frame to do the prediction on.
    :param knn_clf: (optional) a knn classifier object. if not specified, model_save_path must be specified.
    :param model_path: (optional) path to a pickled knn classifier. if not specified, model_save_path must be knn_clf.
    :param distance_threshold: (optional) distance threshold for face classification. the larger it is, the more chance
           of mis-classifying an unknown person as a known one.
    :return: a list of names and face locations for the recognized faces in the image: [(name, bounding box), ...].
        For faces of unrecognized persons, the name 'unknown' will be returned.
    """
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either through knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in
            zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(frame, predictions):
    """
    Shows the face recognition results visually.
    :param frame: frame to show the predictions on
    :param predictions: results of the predict function
    :return opencv suited image to be fitting with cv2.imshow fucntion:
    """
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage


# mongodb implantasyon
cluster = \
    MongoClient("mongodb+srv://seymauguz:QKw5Bpz9Dt%40xam$@cluster0.uhw1fgr.mongodb.net/?retryWrites=true&w=majority")
db = cluster["Babayigit"]
collection = db["Faces"]
'''
MONGODB_HOST = 'localhost'
MONGODB_PORT = 27017
DB_NAME = 'Babayigit'
COLLECTION_NAME = 'Faces'
connection = MongoClient(MONGODB_HOST, MONGODB_PORT)
collection = connection[DB_NAME][COLLECTION_NAME]
projects = collection.find()
'''

# finish here
# new face added check
load_images_from_folder('../face_recognition/to-be-read')
read_from_dv()

"""
item_1 = {
"name": "hasan",
"Face_map": "123,123,123,12,3,123,123,123,12,31,3,12,3,12,33,123,12,312,3,123,12,3,12,312,3,12,3,123,123,,123,,123,12"
}
collection.insert_one(item_1)
connection.close()
"""


if __name__ == "__main__":
    print("Training KNN classifier...")

    classifier = train("../face_recognition/knn_examples/train",
                       model_save_path="../face_recognition/knn_examples/trained_knn_model.clf",
                       n_neighbors=2)
    print("Training complete!")
    # process one frame in every 30 frames for speed
    process_this_frame = 29
    print('Setting cameras up...')
    # multiple cameras can be used with the format url = 'http://username:password@camera_ip:port'
    cap = cv2.VideoCapture(sys.argv[1])
    while True:
        ret, frame = cap.read()
        if ret:
            # Different resizing options can be chosen based on desired program runtime.
            # Image resizing for more stable streaming
            img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            process_this_frame = process_this_frame + 1
            if process_this_frame % 30 == 0:
                predictions = predict(img, model_path="trained_knn_model.clf")
            frame = show_prediction_labels_on_image(frame, predictions)
            cv2.imwrite(f'../face_recognition/images/image_{n}.png', frame)
            n = n + 1
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            cv2.imshow('camera', frame)
            if ord('q') == cv2.waitKey(10):
                cap.release()
                cv2.destroyAllWindows()
                exit(0)
