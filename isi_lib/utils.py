import glob
import os
import operator
import warnings
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFdr
from sklearn.grid_search import GridSearchCV
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from wndcharm.FeatureVector import FeatureVector

# reset numpy error setting since wndcharm messes with it internally
import numpy as np
np.seterr(all='warn')

TRAINING_DATA_DIR = "data"
METADATA_FILE = "metadata_table_all.csv"
TMP_TARGET_TIFF = 'tmp.tiff'


def get_trained_model(source_file_name, classifier='svc', cached=True):
    # load metadata
    metadata = pd.read_csv("/".join([TRAINING_DATA_DIR, METADATA_FILE]), index_col=0)

    # get metadata from file name
    species, development, magnification, probes = get_image_metadata(metadata, source_file_name)

    probe_str = "_".join(sorted(probes))

    train_dir = "/".join(
        [
            TRAINING_DATA_DIR,
            species,
            development,
            magnification,
            probe_str
        ]
    )

    cached_model_file = ".".join([classifier, 'fit'])
    cached_model_path = "/".join([train_dir, cached_model_file])

    # load training model from cached model if cached it  True,
    # else build a new one from training data
    if cached:
        try:
            trained_model = pickle.load(open(cached_model_path, 'rb'))
            class_map = get_class_map(train_dir)
        except Exception:
            print "Failed to load cached training model, building a new one..."
            trained_model = None
            class_map = None
    else:
        trained_model = None
        class_map = None

    if trained_model is None:
        training_data, class_map = load_training_data(train_dir)
        print "Loaded training data"

        trained_model = build_trained_model(training_data, classifier=classifier)
        print "Loaded trained model"

        pickle.dump(trained_model, open(cached_model_path, 'wb'))

    return trained_model, class_map


def predict(region, trained_model, class_map):
    # extract wnd-charm features from region
    target_features = get_target_features(region)

    # classify target features using training data
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        target_prediction = trained_model.predict(target_features)
        try:
            target_prob = trained_model.predict_proba(target_features)
        except Exception:  # TODO: check which exception is raised
            target_prob = None

    if target_prob is not None:
        probabilities = {}
        for i, prob in enumerate(target_prob[0]):
            probabilities[class_map[i + 1]] = prob

        sorted_probabilities = sorted(probabilities.items(), key=operator.itemgetter(1), reverse=True)
    else:
        sorted_probabilities = None

    return class_map[target_prediction[0]], sorted_probabilities


def get_image_metadata(metadata, source_file_name):
    matches = metadata[metadata.img_file == source_file_name]

    first = matches[matches.index == matches.first_valid_index()]

    species = first.organism_label.get_values()[0]
    if species == 'mus musculus':
        species = 'mouse'
    else:
        species = 'human'

    development = first.age_label.get_values()[0]
    magnification = first.magnification.get_values()[0]

    probes = matches.probe_label.unique()
    probes = [p.lower() for p in probes]

    return species, development, magnification, probes


def get_class_map(train_dir):
    class_paths = glob.glob("/".join([train_dir, "*"]))
    class_map = {}

    for i, class_path in enumerate(sorted(class_paths)):
        folder, class_name = os.path.split(class_path)

        class_id = i + 1
        class_map[class_id] = class_name

    return class_map


def load_training_data(train_dir):
    class_paths = glob.glob("/".join([train_dir, "*"]))
    class_map = {}

    ss = []

    for i, class_path in enumerate(sorted(class_paths)):
        folder, class_name = os.path.split(class_path)

        class_id = i + 1
        class_map[class_id] = class_name

        train_files = glob.glob("/".join([class_path, "*.sig"]))

        for f in train_files:
            with open(f) as f_in:
                lines = f_in.readlines()

                values = []
                features = []

                for line in lines[2:]:
                    val, feature = line.split('\t')
                    values.append(val)
                    features.append(feature.strip())

                features.extend(['class_id', 'class', 'Path'])
                values.extend([class_id, class_name, f])

                s = pd.Series(values, index=features)
                ss.append(s)

    training_data = pd.concat(ss, axis=1).T

    # need to convert values to numeric else they will all be 'object' dtypes
    training_data = training_data.convert_objects(convert_dates=False, convert_numeric=True)

    return training_data, class_map


def get_target_features(region):
    region.save(TMP_TARGET_TIFF)

    target_fv = FeatureVector(name='FromTiff', long=True, color=True, source_filepath=TMP_TARGET_TIFF)
    target_fv.GenerateFeatures(quiet=True, write_to_disk=False)

    target_features = pd.Series(target_fv.values, index=target_fv.feature_names)

    return target_features


def build_trained_model(training_data, classifier='svc'):
    alpha = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    ridge_params = {'alpha': alpha}

    c_s = [0.01, 0.1, 1.0, 10.0, 100.0]
    gamma = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    svc_params = [{'kernel': ['rbf'], 'gamma': gamma, 'C': c_s},
                  {'kernel': ['linear'], 'C': c_s}]

    if classifier == 'svc':
        clf = GridSearchCV(SVC(probability=True), svc_params, cv=5)
    elif classifier == 'ridge':
        clf = GridSearchCV(RidgeClassifier(), ridge_params, cv=5)
    else:
        raise NotImplementedError("Only 'svc' (default) and 'ridge' classifiers are supported")

    pipe = Pipeline([
        ('standard_scalar', StandardScaler()),
        ('feature_selection', SelectFdr()),
        ('classification', clf)
    ])

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        pipe.fit(
            training_data.ix[:, :-3],
            training_data.ix[:, -3].astype('int')
        )

    return pipe
