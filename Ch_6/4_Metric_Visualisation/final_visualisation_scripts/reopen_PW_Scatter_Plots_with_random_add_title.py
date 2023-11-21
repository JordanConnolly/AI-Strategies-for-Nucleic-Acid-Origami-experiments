import matplotlib.pyplot as plt
from textwrap import wrap

experiment_list = [
    "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_10",
    "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_25",
    "RANDOM_FORWARD_CXPB_SWEEP_12_POINT_POP_CXPB_90_MUTATIONS_16BP_INDMUTATION_50",
]

experiment_list_name = ["Genetic Algorithm Sweep Set 1", "Genetic Algorithm Sweep Set 2",
                        "Genetic Algorithm Sweep Set 3"]

list_of_origami_names = ["hj", "nanoribbonRNA", "fourfinger-linear",
                         "fourfinger-circular", "minitri", "DBS_square",
                         "Abrick", "6hb", "ball"]

list_of_origami_names_for_plot = ["Single HJ",
                                  "Nanoribbon (RNA)",
                                  "M1.3 Four Finger (Linear)",
                                  "M1.3 Four Finger (Circular)",
                                  "Mini Triangle",
                                  "Jurek Square",
                                  "Solid Brick",
                                  "6 Helix Bundle",
                                  "Ball"]

for i in range(len(list_of_origami_names)):
    # set the origami for plotting
    origami_name = list_of_origami_names[i]
    origami_name_for_plot = list_of_origami_names_for_plot[i]

    for j in range(len(experiment_list)):
        experiment = experiment_list[j]
        experiment_name_for_plot = experiment_list_name[j]

        url = r"Plot_Output/" + origami_name + "_" + experiment + "_pairwise_scatter_plot.png"
        pic = plt.imread(url)

        print(pic.shape)

        # display the image in a mpl figure
        plt.figure(figsize=(8, 8))

        # load image
        plt.imshow(pic)

        title = ("\n".join(wrap("Pairwise Plot for " + origami_name_for_plot + " " + experiment_name_for_plot, 90)))
        # modify image title
        plt.title(title, fontsize=12)
        plt.axis('off')
        # show image
        plt.savefig("Plot_Output/" "titled_pairwise_scatter_plot_for_" + origami_name + "_" + experiment + ".png",
                    bbox_inches='tight', format='png')
        plt.close()
        plt.cla()


