#!/usr/bin/python
import mistune
import re

renderer = mistune.Renderer(escape=True, hard_wrap=True)
# use this renderer instance
markdown = mistune.Markdown(renderer=renderer)

# input_text = """AAAATTAAAATTACCTGAGCAAAAGAAGATGATG
# AGCCGGCGAACGTGGCAACCACCA
# GAAAGCGAAAGGAGCGGGCGCAAACAATAACG
# GATTCGCCTGATTGCCAGTAACA
# GCCAGTTAGTCCTGAACAAGAAAAATAATAT*[Poly-A]//Non-binding-nucleotides//*AAAAAAAAAA
# CGCTAACGAATTTACGAGCATGTAGAAACC*[Poly-A]//Non-binding-nucleotides//*AAAAAAAAAAAA
# CCAGCTACTCGGCTGTCTTTCCTTATCATTCC*[Poly-A]//Non-binding-nucleotides//*AAAAAAAAAA"""

input_text = """
(staple_86)ATTACCTTCAAAATCGCGCAGAGAGATTTTC(length: 31)
(staple_87)AAAATTAAAATTACCTGAGCAAAAGAAGATGATG(length: 34)
(staple_88)AGCCGGCGAACGTGGCAACCACCA(length: 24)
(staple_89)GAAAGCGAAAGGAGCGGGCGCAAACAATAACG(length: 32)
(staple_90)GATTCGCCTGATTGCCAGTAACA(length: 23)
(test_left)AAAAAAAAAA*[Poly-A]//Non-binding-nucleotides//*CGCTAACGAATTTACGAGCATGTAGAAACC(length: 41)
(test_mid) CGCTAACGAATTTACGAGCAT*[Poly-A]//Non-binding-nucleotides//*AAAAAAAAAAAAGTAGAAACC(length: 42)
(test_mid_left) AAAAAAAAAAAACGCTAAC*[Poly-A]//Non-binding-nucleotides//*GAATTTACGAGCATGTAGAAACC(length: 42)
(staple_93)CCAGCTACTCGGCTGTCTTTCCTTATCATTCC*[Poly-A]//Non-binding-nucleotides//*AAAAAAAAAA(length: 42)
(staple_94)CCTTAAAGTATTAAACCAAGTACCGCACTC*[Poly-A]//Non-binding-nucleotides//*AAAAAAAAAA(length: 40)
(staple_95)GCGAACCTCAAGCAAGCCGTTTTTATTTTCAT*[Poly-A]//Non-binding-nucleotides//*AAAAAAAAAA(length: 42)"""


def add_header(text):
    add_header_short = """
| First Column | Second Column | Third Column | Fourth Column |
| ------------ | ------------- | ------------ | ------------- |
"""

    add_header_long = """
| Staple Number | Core Sequence | Modification Name | Modification Type | Modification | Sequence Length | Raw Sequence  |
| ------------- | ------------- | ----------------- | ----------------- | ------------ | --------------- |-------------- |
"""

    headed_text = add_header_long + text
    return headed_text


def create_markdown_table(marked_sequence):
    # create a markdown table replacing the marked-up tags with pipe: |
    # remove entire set of tags
    # raw_sequence = re.sub(r'\*[^*]*\*', ' | ', marked_sequence)

    # remove tags or replace with pipes
    piped_asterix_sequences = marked_sequence.replace("*", "")
    piped_forward_slash_sequences = piped_asterix_sequences.replace("//", " | ")
    piped_square_bracket_sequences = piped_forward_slash_sequences.replace("[", " | ")
    piped_remove_extra_square_bracket_sequences = piped_square_bracket_sequences.replace("]", "")
    piped_round_bracket_sequences = piped_remove_extra_square_bracket_sequences.replace("(", " | ")
    piped_sequences = piped_round_bracket_sequences.replace(")", " | ")
    return piped_sequences


def fill_in_raw_seq_fields(text):
    list_of_lines = []
    pipe = " | "
    for line in text.splitlines():
        # counts number of pipes in each line
        count = line.count("|")

        if count == 4:

            # fixes the raw sequences

            split_line = line.split("|")
            new_line = pipe + split_line[1] + pipe + split_line[2] + ("| N/A | N/A | N/A | ") + split_line[3] + \
                       pipe + split_line[2] + pipe
            list_of_lines.append(new_line)

        elif count == 7:

            # works for modified sequences with modifications on the right hand side

            split_line = line.split("|")
            # retrieve full sequence
            full_seq = split_line[2].replace(" ", "") + split_line[5].replace(" ", "") + pipe
            line = line + full_seq
            list_of_lines.append(line)

        else:

            # ensures header is kept the same

            list_of_lines.append(line)

    # creates a multi-line to keep formatting
    return_text = """
    {}
    """.format("\n".join(list_of_lines))
    return return_text


def parse_markdown(text):
    # used to parse the marked-up sequences text
    # we will store the rendered html as html_str
    html_str = markdown(text=text)

    # OPTIONAL: replace html table with CSS table for style viewing
    html_str = html_str.replace("<table>", "<table class=\"styled-table\">")
    return html_str


def store_markdown_html(text):
    html_str_to_store = text
    # create pre-defined HTML file
    html_file = open("html_files/file_test.html", "w")
    # pre-header to define html file and style
    pre_header_str = """
    <!DOCTYPE html>
    <html>
    <link rel="stylesheet" type="text/css" href="origami_sequence_style_sheet.css" />
    """
    html_file.write(pre_header_str)
    html_file.close()

    # create table in HTML file
    html_file_append = open("html_files/file_test.html", "a")
    # actual table content appended to written html file
    html_file_append.write(html_str_to_store)
    html_file_append.close()
    return


# add header text
input_text = add_header(input_text)
# replace marked-up with markdown pipes and hyphen headers
table_formatted_text = create_markdown_table(input_text)
# fill in raw sequence blanks
markdown_table_pre_parsed = fill_in_raw_seq_fields(table_formatted_text)
# use mistune parser to form html format and create a saved file
parsed = parse_markdown(markdown_table_pre_parsed)
# store the content onto that saved file
store_markdown_html(parsed)

