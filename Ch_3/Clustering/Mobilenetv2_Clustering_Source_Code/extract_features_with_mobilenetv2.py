import os
import glob
import numpy as np
import pandas as pd
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from PIL import Image


def load_data(file_path):
    return pd.read_csv(file_path)


def extract_features(img_path, model):
    img = Image.open(img_path).resize((224, 224))
    img = img.convert("RGB")
    img_array = np.array(img)
    x = img_array.reshape(224, 224, 3)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return model.predict(x).flatten()


def main():
    data_set = load_data("../Jordan_Connolly-nucleic-acid-origami-database/Full_Data_Set/Machine-Learning-Sets/"
                         "All_Papers_Non_RevNano_Feature_Subset.csv")
    df = data_set.groupby("Paper Number").filter(lambda x: len(x) == 1)
    paper_number_list = df["Paper Number"].to_list()

    IMG_DIR = 'E:/PhD Files/RQ1/Investigation_into_feasibility_of_image_work/images_that_are_less_noisy/'
    model = MobileNetV2(weights='imagenet', include_top=False)

    features = []
    image_paper_list = []
    for img_path in glob.glob(os.path.join(IMG_DIR, '*')):
        filename = os.path.basename(img_path)
        value = int(filename.split('_')[2])

        if value not in paper_number_list:
            print(value)
            continue

        feature = extract_features(img_path, model)
        features.append(feature)
        image_paper_list.append(value)

    print("Shape of feature array:", np.array(features).shape)

    image_feature_df = pd.DataFrame.from_records(features)
    image_feature_df["Paper Number"] = image_paper_list
    image_feature_df.to_csv("unmerged_mobilenetv2_image_features_no_duplicates.csv")

    image_paper_list_df = pd.DataFrame(image_paper_list)
    image_paper_list_df.to_csv("image_paper_number.csv")

    df.merge(image_feature_df, on="Paper Number")
    image_feature_df.to_csv("mobilenetv2_image_features_no_duplicates.csv")


if __name__ == "__main__":
    main()
