import glob
import os
import subprocess
import imageio
import re
import numpy as np
import pandas as pd
from os import path
import re
import glob
from textwrap import wrap


# access and import the details for the plot from a table
basepath = path.dirname(__file__)
# access and import the details for the plot from a table
filepath = path.abspath(path.join(basepath, "..", "GA_Raw_Results_Analysis", "Analysis_Output", "CXPB_SWEEP_SET_2_hypervolume_table_select.csv"))
table_df = pd.read_csv(filepath)

# experiment_list = [26, 17]
# for experiment in experiment_list:
# for experiment in table_df.itertuples():
#     rep = experiment[2]
#     experiment_name = experiment[7]
#     print(experiment_name, rep)
    # name of experiment file
    # experiment_name = "CXPB_SWEEP_CXPB_30_MUTATIONS_16BP_INDMUTATION_25"

for experiment in table_df.itertuples():
    rep = experiment[3]
    experiment_name = experiment[2]
    if rep > 0:
        print(experiment_name)

        # make sure that the sort works numerically
        numbers = re.compile(r'(\d+)')


        def numerical_sort(value):
            parts = numbers.split(value)
            parts[1::2] = map(int, parts[1::2])
            return parts


        experiment = "experiment_1/"
        images_to_video_path = "output_data/" + experiment
        output_path = "2d_gif_output_data/"


        def create_2d_front_gif(fronts):
            images = []
            filenames = []
            front_name = "pareto_front_" + str(fronts)
            for file_name in sorted(glob.glob(images_to_video_path + "*" + "Repetition " + str(rep) + "_*" +
                                              front_name +
                                              "*" + experiment_name + ".png"), key=numerical_sort):
                filenames.append(file_name)
            # create the gif
            movie_title = front_name + "_all_movie"
            for filename in filenames:
                images.append(imageio.imread(filename))
            imageio.mimsave(output_path + movie_title + '.gif', images, duration=0.25)
            return


        create_2d_front_gif(fronts=1)
        create_2d_front_gif(fronts=2)
        create_2d_front_gif(fronts=3)


        def create_horizontal_gifs(input_path):
            # combine GIFs into one horizontal gif
            gif1 = imageio.get_reader(input_path + 'pareto_front_1_all_movie.gif')
            gif2 = imageio.get_reader(input_path + 'pareto_front_2_all_movie.gif')
            gif3 = imageio.get_reader(input_path + 'pareto_front_3_all_movie.gif')
            number_of_frames = min(gif1.get_length(), gif2.get_length(), gif3.get_length())
            print(number_of_frames)
            number_of_frames_1 = gif1.get_length()
            number_of_frames_2 = gif2.get_length()
            number_of_frames_3 = gif3.get_length()
            print(number_of_frames_1, number_of_frames_2, number_of_frames_3)

            new_gif = imageio.get_writer(input_path + experiment_name +
                                         'new_pareto_front_all_combined_movie'
                                         + "repetition_" + str(rep) + '.gif', duration=0.25)
            for frame_number in range(number_of_frames):
                img1 = gif1.get_next_data()
                img2 = gif2.get_next_data()
                img3 = gif3.get_next_data()
                new_image = np.hstack((img1, img2, img3))
                new_gif.append_data(new_image)
            gif1.close()
            gif2.close()
            gif3.close()
            new_gif.close()

        create_horizontal_gifs(input_path=output_path)
