# Source: "https://github.com/UC-Davis-molecular-computing/
# Source:  scadnano-python-package/blob/master/tutorial/tutorial.md#create-empty-design"


import scadnano as sc
import xlwt


max_offset = 288
max_offset_halved = int(max_offset/2)
staple_length = 32
staple_length_halved = int(staple_length/2)
helix_number = 24


def main() -> None:
    design = create_design()
    design.write_scadnano_file()
    design.write_idt_plate_excel_file()


def create_design() -> sc.Design:
    helices = [sc.Helix(max_offset=max_offset) for _ in range(helix_number)]
    design = sc.Design(helices=helices, grid=sc.square)

    add_scaffold_precursors(design)
    add_scaffold_crossovers(design)

    add_staple_precursors(design)
    add_staple_crossovers(design)
    # add_staple_nicks(design)

    add_twist_correction_deletions(design)
    design.assign_m13_to_scaffold()

    return design


def add_scaffold_precursors(design: sc.Design) -> None:
    for helix in range(0, helix_number-1, 2):  # scaffold goes forward on even helices
        design.strand(helix, 0).move(max_offset).as_scaffold()
    for helix in range(1, helix_number-1, 2):  # scaffold goes reverse on odd helices
        design.strand(helix, max_offset).move(-max_offset).as_scaffold()
    design.strand(helix_number-1, max_offset).move(-max_offset_halved).as_scaffold()  # bottom part of scaffold has a "nick"
    design.strand(helix_number-1, max_offset_halved).move(-max_offset_halved).as_scaffold()  #


def add_scaffold_crossovers(design: sc.Design) -> None:
    for helix in range(1, helix_number-1, 2):  # scaffold interior crossovers
        design.add_full_crossover(helix=helix, helix2=helix + 1, offset=max_offset_halved, forward=False)

    for helix in range(0, helix_number-1, 2):  # scaffold edges crossovers
        design.add_half_crossover(helix=helix, helix2=helix + 1, offset=0, forward=True)
        design.add_half_crossover(helix=helix, helix2=helix + 1, offset=max_offset - 1, forward=True)


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


def add_staple_nicks(design: sc.Design) -> None:
    for helix in range(helix_number):
        start_offset = helix_number if helix % 2 == 0 else 40
        for offset in range(start_offset, (max_offset - staple_length_halved), staple_length):
            design.add_nick(helix, offset, forward=helix % 2 == 1)


def add_twist_correction_deletions(design: sc.Design) -> None:
    for helix in range(24):
        for offset in range(19, 286, 48):
            design.add_deletion(helix, offset)


if __name__ == '__main__':
    main()
