import scadnano as sc
import scadnano.modifications as mod


def create_design():
    # helices
    helices = [sc.Helix(max_offset=48), sc.Helix(max_offset=48)]

    # whole design
    design = sc.Design(helices=helices, grid=sc.square)

    # for absolute offsets, call method "to"
    # left staple
    design.strand(1, 8).to(24).cross(0).to(8)

    # for relative offsets, call method "move"
    # right staple
    design.strand(0, 40).move(-16).cross(1).move(16).with_modification_5p(mod.biotin_5p)

    # scaffold
    design.strand(1, 24).move(-16).cross(0).move(32).loopout(1, 3).move(-16).as_scaffold()

    # deletions and insertions added to design are added to both strands on a helix
    design.add_deletion(helix=1, offset=20)
    design.add_insertion(helix=0, offset=14, length=1)
    design.add_insertion(helix=0, offset=26, length=2)

    # also assigns complement to strands other than scaf bound to it
    design.assign_dna(design.scaffold, 'AACGT' * 18)

    return design


if __name__ == '__main__':
    design = create_design()
    design.write_scadnano_file(directory='output_designs')
