import glob
import os
import warnings
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFdr
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import RidgeClassifier
from sklearn.pipeline import Pipeline
from wndcharm.FeatureVector import FeatureVector

# reset numpy error setting since wndcharm messes with it internally
import numpy as np
np.seterr(all='warn')

TRAINING_DATA_DIR = "data"
METADATA_FILE = "metadata_table_all.csv"
TMP_TARGET_TIFF = 'tmp.tiff'


def predict(region, source_file_name):
    # load metadata
    metadata = pd.read_csv("/".join([TRAINING_DATA_DIR, METADATA_FILE]), index_col=0)

    # get metadata from file name
    species, development, magnification, probes = get_image_metadata(metadata, source_file_name)

    # load training data
    training_data, class_map = load_training_data(
        species,
        development,
        magnification,
        probes
    )

    # extract wnd-charm features from region
    target_features = get_target_features(region)

    # classify target features using training data
    target_class = classify(training_data, target_features)

    # return class name and confidence
    return class_map[target_class]


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


def load_training_data(species, development, magnification, probes):
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

    class_paths = glob.glob("/".join([train_dir, "*"]))
    class_map = {}

    ss = []

    for i, class_path in enumerate(class_paths):
        folder, class_name = os.path.split(class_path)

        class_id = i + 1
        class_map[class_id] = class_name

        train_files = glob.glob("/".join([class_path, "*.sig"]))

        for f in train_files:
            with open(f) as f_in:
                lines = f_in.readlines()

                vals = []
                features = []

                for line in lines[2:]:
                    val, feature = line.split('\t')
                    vals.append(val)
                    features.append(feature.strip())

                features.extend(['class_id', 'class', 'Path'])
                vals.extend([class_id, class_name, f])

                s = pd.Series(vals, index=features)
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


def classify(training_data, target_features):
    alpha = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    ridge_params = {'alpha': alpha}

    clf = GridSearchCV(RidgeClassifier(), ridge_params, cv=5)

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

        target_prediction = pipe.predict(target_features)

    return target_prediction[0]
