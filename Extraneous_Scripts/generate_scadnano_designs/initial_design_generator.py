import scadnano as sc
import xlwt
import time

# /////////////////////////////////
# /// ORIGAMI DESIGN PARAMETERS ///
# /////////////////////////////////

# Initial Parameters for design
helix_number = 0  # (WIDTH)  # Begin at 1, Increase by +1
max_offset = 0  # (LENGTH)  # Begin at 36, Increase by +9
max_offset_halved = int(max_offset / 2)
staple_length = 0  # global staple length; Begin at 8, increase by +2
staple_length_halved = int(staple_length / 2)
add_twist_correction = True

# design storage location
storage_path = "output_designs/"
design_storage = storage_path + "design/"
sequence_storage = storage_path + "seqs/"


# /////////////////////////////////////
# /// Generate Rectangles / Squares ///
# /////////////////////////////////////


def create_simple_shapes():
    """
    Create simple shapes that vary in helix number (length) and max_offset (width)
    They are simple because they do not vary their offset (nucleotides per helix).

    Examples synthesised here:
    Helix Bundle (large max_offset (width) with numerous helices)
    Rectangle (longer sides than width; vice-versa).
    Square Tile (equilateral in width and length).

    Alternatives not synthesised here:
    An example of an offset shape would be a triangle or planar circle.
    An example of an offset shape with gaps (breaks in helix) would be a rothemund smiley face.
    An example of an non-offset shape with gaps would be a rectangle used as a frame or template.
    An example of an non-offset twisted helix bundle.
    An example of an offset twisted shape.
    """

    # Set Upper Bounds of Synthetic Origami
    helix_upper_bound = 22
    helix_lower_bound = 16
    upper_bound_offset = 288  # 288
    lower_bound_offset = 64  # 64
    upper_bound_staple_length = 56  # works best in multiples of 8
    lower_bound_staple_length = 16  # works best in multiple of 8
    design_counter = 0
    time_delay = 0

    # Global Variables
    global helix_number  # (WIDTH)  # Begin at 1, Increase by +2
    global max_offset  # (LENGTH)  # Begin at 36, Increase by +8
    global max_offset_halved
    global staple_length
    global staple_length_halved

    # Create lists of feasible parameter ranges
    helix_range = [i for i in range(helix_lower_bound, helix_upper_bound+1) if i % 2 == 0]
    print(helix_range)

    # creates a list of max_offset values that are divisible by 9
    max_offset_range = [i for i in range(lower_bound_offset, upper_bound_offset + 1) if i % 1 == 0]
    # print("max_offset:", max_offset)

    # creates a list of staple lengths to use that are between 8 and upper bound in multiples of 8
    staple_length_range = [i for i in range(lower_bound_staple_length + helices,
                                            upper_bound_staple_length + 1) if i % 8 == 0]
    print(staple_length_range)

    # Iterate over possible design parameters
    for helices in helix_range:
        helix_number = helices

        for max_offset_value in max_offset_range:
            max_offset = max_offset_value
            max_offset_halved = int(max_offset/2)

            for staple_length_value in staple_length_range:
                staple_length = staple_length_value

                # print("staple_length:", staple_length)

                # Check outer staple length against max_offset
                if staple_length < max_offset:
                    staple_length_halved = int(staple_length / 2)

                    # Call scad-nano to design
                    design = create_design("original")

                    # if helices < 16:
                    #     design = create_design("simple")
                    # if helices >= 16:
                    #     design = create_design("over_16_helices")

                    # Iterate over staple strands and add to list
                    strand_len_list = []
                    for strand in design.strands:
                        if not strand.is_scaffold:
                            individual_staple_len = len(strand.dna_sequence)
                            strand_len_list.append(individual_staple_len)

                    # Iterate over staple strands and check sizes
                    if all(i >= 5 for i in strand_len_list) is False:
                        print("strand less than 5 nucleotides")

                    if all(i >= 5 for i in strand_len_list) is True:
                        if all(i <= (staple_length + 32) for i in strand_len_list) is True:
                            design_counter += 1
                            print("current design iter:", design_counter, "; helices:", str(helix_number),
                                  "; width:", str(max_offset), "; staple length:", str(staple_length))
                            file_name = "break_scadnano_design_" + str(design_counter)
                            # design.write_scadnano_file(directory=design_storage, filename=file_name + ".sc")
                            # design.write_idt_plate_excel_file(directory=sequence_storage, filename=file_name + ".xls")

                            # set a delay to debug
                            time.sleep(time_delay)

    return helix_number, max_offset, staple_length, max_offset_halved, staple_length_halved


def create_design(type_of_structure) -> sc.Design:
    helices = [sc.Helix(max_offset=max_offset) for _ in range(helix_number)]
    design = sc.Design(helices=helices, grid=sc.square)

    add_scaffold_precursors(design)
    add_scaffold_crossovers(design)

    add_staple_precursors(design)
    add_staple_crossovers(design)

    if type_of_structure is "original":
        add_staple_nicks_original(design)
    if type_of_structure is "simple":
        add_staple_nicks_simple(design)
    if type_of_structure is "over_16_helices":
        add_staple_nicks_over_16_helices(design)

    add_twist_correction_deletions(design)
    design.assign_m13_to_scaffold()
    # synthetic scaffold
    # design.assign_dna()
    # SOURCE: https://scadnano-python-package.readthedocs.io/en/latest/#scadnano.Design.assign_dna
    return design


def add_scaffold_precursors(design: sc.Design) -> None:
    for helix in range(0, helix_number-1, 2):  # scaffold goes forward on even helices
        design.strand(helix, 0).move(max_offset).as_scaffold()
    for helix in range(1, helix_number-1, 2):  # scaffold goes reverse on odd helices
        design.strand(helix, max_offset).move(-max_offset).as_scaffold()
    design.strand(helix_number-1, max_offset).move(-max_offset_halved).as_scaffold()  # bottom scaffold part has a nick
    design.strand(helix_number-1, max_offset_halved).move(-max_offset_halved).as_scaffold()  #


def add_scaffold_crossovers(design: sc.Design) -> None:
    for helix in range(1, helix_number-1, 2):  # scaffold interior crossovers
        design.add_full_crossover(helix=helix, helix2=helix + 1, offset=max_offset_halved, forward=False)
    for helix in range(0, helix_number-1, 2):  # scaffold edges crossovers
        design.add_half_crossover(helix=helix, helix2=helix + 1, offset=0, forward=True)
        design.add_half_crossover(helix=helix, helix2=helix + 1, offset=max_offset-1, forward=True)


def add_staple_precursors(design: sc.Design) -> None:
    staples = [sc.Strand([sc.Domain(helix=helix, forward=helix % 2 == 1, start=0, end=max_offset)])  # noqa
               for helix in range(helix_number)]
    for staple in staples:
        design.add_strand(staple)


def add_staple_crossovers(design: sc.Design) -> None:
    for helix in range(helix_number-1):
        start_offset = staple_length_halved if helix % 2 == 0 else staple_length
        for offset in range(start_offset, max_offset, staple_length):
            if offset != max_offset_halved:  # skip crossover near seam
                design.add_full_crossover(helix=helix, helix2=helix + 1, offset=offset,
                                          forward=helix % 2 == 1)


def add_staple_nicks_original(design: sc.Design) -> None:
    for helix in range(helix_number):
        start_offset = helix_number if helix % 2 == 0 else (staple_length + int(staple_length/2) + 2)

        # print(helix)  # debug
        # for offset in range(int(start_offset/2), (max_offset - int(staple_length/2)), staple_length):

        for offset in range(int(start_offset/2),
                            (max_offset - int(staple_length / 4)), staple_length):

            # print("offset:", offset, ";start offset:", start_offset, ";max offset:",  # debug
            #       (max_offset - int(staple_length/2)), ";staple length:", staple_length)  # debug

            design.add_nick(helix, offset, forward=helix % 2 == 1)


def add_staple_nicks_simple(design: sc.Design) -> None:
    for helix in range(helix_number):
        start_offset = helix_number if helix % 2 == 0 else staple_length - helix_number
        for offset in range(start_offset, (max_offset - helix_number), staple_length):
            design.add_nick(helix, offset, forward=helix % 2 == 1)


def add_staple_nicks_over_16_helices(design: sc.Design) -> None:
    for helix in range(helix_number):
        start_offset = helix_number if helix % 2 == 0 else staple_length - helix_number
        for offset in range(start_offset, (max_offset - helix_number), staple_length):
            # print(start_offset, max_offset, staple_length)
            # print(helix, offset)
            design.add_nick(helix, offset - 3, forward=helix % 2 == 1)


def add_staple_nicks_varying_offsets(design: sc.Design) -> None:
    for helix in range(helix_number):
        start_offset = helix_number if helix % 2 == 0 else staple_length - helix_number
        for offset in range(start_offset, (max_offset - helix_number), staple_length):
            design.add_nick(helix, offset, forward=helix % 2 == 1)


# Twist correction adds deletions to HELIX not staples.
def add_twist_correction_deletions(design: sc.Design) -> None:
    if add_twist_correction is True:
        for helix in range(helix_number):
            for offset in range(19, max_offset-2, 48):
                design.add_deletion(helix, offset)
    else:
        return


create_simple_shapes()

if __name__ == "__main__":
    create_simple_shapes()
