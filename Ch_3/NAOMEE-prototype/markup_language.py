"""
////////////////////////
// Syntax and Grammar //
////////////////////////
What are we labelling, based upon paper examples:

::: Staple Sequence :::
plain_text staples are modified at certain points? make bold using **
plain_text staple is entirely modified? make all bold using **

::: Staple Modification :::
HairpinLoop//Sequence - is sequence a loop or creates a loop?
Overhang//Sequence - is part of sequence an overhang, non-binding?
[Probe-Name]//Probe//Sequence - is sequence used as a probe?
Thiolation//Sequence - is sequence thiolated?
[Joins-What-Name]//Aptamer-or-capture//Sequence - is sequence an aptamer linker or capture strand?
[PolyA-Name]//Non-Binding-Nucleotides//Sequence - is sequence used to block binding / purposefully not bind scaffold?

::: Scaffold Modification :::
Parse where scaffold is cut or unused versus used in folding. Needs editing and further thought
plain_text scaffold
///////////////////////////
// Parsing of Plain-Text //
///////////////////////////
1) Read in plain text document and edit it using mark up to create new mark up file
"""

import re
import sys
import tkinter as tk
from tkinter import END
from tkinter import messagebox as mb
from tkinter import filedialog
from tkinter import font
from tkinter import ttk
from convert_to_html import add_header, create_markdown_table, fill_in_raw_seq_fields, \
    parse_markdown, store_markdown_html


print("current location (sys.argv):", sys.argv)

'''
Set the global variables in the script for later usage
'''

# set global variable for open file name
global open_file_name_global
open_file_name_global = False

# set global variable for selected modification use
global myLabel_1_global
myLabel_1_global = False

# set global variable for selected modification name
global myLabel_2_global
myLabel_2_global = False

'''
Syntax of the Markup Language
# The overriding design goal for Markdown’s formatting syntax is to make it as readable as possible.
# The idea is that a Markdown-formatted document should be publishable as-is, as plain text,
# without looking like it’s been marked up with tags or formatting instructions.
# need untagged and "raw" sequences, that are read as plain text (sequences)
# need to be able to return raw sequences with and without modifications added (folding only sequences)

# https://en.wikipedia.org/wiki/Lightweight_markup_language#Link_syntax

EXAMPLES:
plain_text = "AGCGTAGTACGTACGTACGTACGTTTTTTTTT"  # basic raw text 
marked_modification_in_sequence = "AGCGTAGTACGTACGTACGTACG**TTTTTTTTT"  # basic raw text
modification_tag = "//Non-binding//"  # modification tag example
modification_note_tag = "[Poly-T]"  # modification descriptor example
origami_descriptor_tags = "(staple-57)(length: 42)" # descriptors example

SYNTAX:
marked_modification_in_sequence = "**"  # basic raw text syntax
modification_tag = "////"  # modification tag syntax
modification_note_tag = "[]"  # add a note / name for modification added
origami_descriptor_tags = "()()"  # descriptors syntax
'''

# what is the modification used for?
# preset list of modifications
preset_list_of_modifications = ['Non-binding-nucleotides', 'Loop', 'Hairpin-Loop', 'Aptamer-Linker',
                                'Adsorption', 'Cavity', 'Probe', 'Over-hang']
list_of_modifications = preset_list_of_modifications

# what is the specific modification called?
# extendable list of specific modification names (loaded from txt file)
extendable_list_of_names = ""

# preset list of specific modification names
preset_list_of_names = ['Poly-G', 'Poly-C', 'Poly-T', 'Poly-A', 'Poly-AT', 'Poly-GC', 'Repeating-Nucleic-Acids',
                        'Peptide', 'Thiol-SH', 'Gold', 'Drug',
                        'Cy5', 'Cy3', 'Alexa647', 'Fluorophore', 'SYBR-Gold-Gel-Stain', 'Texas-Red',
                        'PDGF', 'Thrombin', 'Meleimide', 'Adenosine', 'Rhodamine-G', 'Dabcyl', 'Cholesterol',
                        'Biotin', 'GOx-glucose-oxidase', 'HRP-horseradish-peroxidase', 'GOx/HRP']
list_of_names = preset_list_of_names
# 'BHQ1', 'ROX', 'FAM'; extra list of dyes: https://www.leica-microsystems.com/science-lab/fluorescent-dyes/

'''
Mark the raw sequence at certain locations to indicate where sequence tags are added
- You want to be able to add tags between the asterisk markings of where the tags should be placed
'''


def insert_test(selected_text):
    # insert the modification type tag
    # if ** is present, replace with *////*
    my_text = re.sub(r'\*[^*]*\*', '*////*', selected_text)
    return my_text


def insert_modification_tag(raw_sequence, modification_used):
    # insert the modification type tag
    # - if ** is present, replace with *////*
    inserted_modification_tag_markers = re.sub(r'\*[^*]*\*', '*////*', raw_sequence)
    # add more of these inflexible modifications
    if modification_used == 1:
        entry = str(myLabel_1_global)
        # - wherever //// is present, replace with the entry provided and
        removed_modification_name_tag = re.sub(r'\/[^*]*\/', '//' + entry + '//', inserted_modification_tag_markers)
        return removed_modification_name_tag
    else:
        return inserted_modification_tag_markers


def insert_modification_note_tag(raw_sequence):
    # check the sequence contains modification markers
    # - check to see if //// is present
    inserted_modification_tag_markers = re.findall(r'\/[^*]*\/', raw_sequence)
    check_modification_added = len(inserted_modification_tag_markers)
    # insert the modification note tag
    if check_modification_added >= 1:
        entry = "[" + str(myLabel_2_global) + "]"
        # - wherever */ is present, place *[entry]//
        inserted_modification_note = re.sub(r'\*/*\/', '*' + entry + '//', raw_sequence)
        return inserted_modification_note
    else:
        return


'''
Remove the individual tags from marked up sequences
'''


def remove_round_brackets(marked_sequence):
    # remove origami specific tag
    removed_origami_specific_tag = re.sub(r'\([^)]*\)', '', marked_sequence)
    return removed_origami_specific_tag


def remove_square_brackets(marked_sequence):
    # remove modification name tag
    removed_modification_name_tag = re.sub(r'\[[^]]*\]', '', marked_sequence)
    return removed_modification_name_tag


def remove_forward_slash(marked_sequence):
    # remove modification tag
    removed_modification_tag = re.sub(r'\/[^)]*\/', '', marked_sequence)
    return removed_modification_tag


'''
Remove the entire set of tags added to the sequence
'''


def return_raw_sequence_steps(marked_sequence):
    removal_1 = remove_round_brackets(marked_sequence)
    removal_2 = remove_square_brackets(removal_1)
    removal_3 = remove_forward_slash(removal_2)
    return removal_3, marked_sequence


def return_raw_sequence(marked_sequence):
    # remove entire set of tags
    raw_sequence = re.sub(r'\*[^*]*\*', '', marked_sequence)
    raw_sequence = remove_round_brackets(raw_sequence)
    return raw_sequence, marked_sequence


def return_raw_sequence_leave_round_brackets(marked_sequence):
    # remove entire set of tags
    raw_sequence = re.sub(r'\*[^*]*\*', '', marked_sequence)
    return raw_sequence, marked_sequence


'''
Created a simple GUI to showcase the mark up of plain text
'''

root = tk.Tk()
root.title('Sequence Editor')
root.iconbitmap('ICOforMarkdown.ico')
root.geometry("1900x1600")


# create file function
def new_file():
    # delete previous sequences
    my_text.delete("1.0", END)
    # update status on GUI
    root.title(' New File - Sequence Editor')
    status_bar.config(text="New File        ")
    global open_file_name_global
    open_file_name_global = False


def open_file():
    if mb.askyesno('Verify', 'Open new file? This will delete current text'):
        # delete the marked / left panel sequences
        my_text.delete("1.0", END)
        # delete the raw / right panel sequences
        my_text_raw.delete('1.0', 'end-1c')
        # grab filename
        text_file = filedialog.askopenfilename(initialdir="C:/", title="Open File", filetypes=(
            ("Text Files", "*.txt"),
            ("CSV Files", "*.csv"),
            ("All FileTypes", "*.*")))
        if text_file:
            # global file name
            global open_file_name_global
            open_file_name_global = text_file

        # update status on GUI
        name = text_file
        status_bar.config(text=f'{name}')
        name = name.replace("C:/", "")
        root.title(f'{name} - Sequence Editor')

        # open the file
        text_file = open(text_file, 'r')
        stuff = text_file.read()
        # add file to text box
        my_text.insert(END, stuff)
        # close the opened file
        text_file.close()
    else:
        pass


def save_as_file():
    # save as file
    text_file = filedialog.asksaveasfilename(defaultextension=".*", initialdir="C:/", title="Save File",
                                             filetypes=(("Text Files", "*.txt"),
                                                        ("CSV Files", "*.csv"),
                                                        ("All FileTypes", "*.*")))
    if text_file:
        # update the status in GUI
        name = text_file
        status_bar.config(text=f'{name}')
        name = name.replace("C:/", "")
        root.title(f'{name} - Sequence Editor')

        # save the file
        text_file = open(text_file, 'w')
        text_file.write(my_text.get(1.0, END))
        # close the file
        text_file.close()


def save_file():
    global open_file_name_global
    if open_file_name_global:
        # save the file
        text_file = open(open_file_name_global, 'w')
        text_file.write(my_text.get(1.0, END))
        # close the file
        text_file.close()
        # pop-up box code
        status_bar.config(text=f'Saved: {open_file_name_global}')
    else:
        save_as_file()


def callback():
    # quitting callback, produces Question about QUITTING program
    if mb.askyesno('Verify', 'Really quit?'):
        root.quit()
    else:
        mb.showinfo('No', 'Quit has been cancelled')


def select_all():
    # add sel tag to select all text
    my_text.tag_add('sel', '1.0', 'END')


def clear_all():
    # clear all text with delete
    my_text.delete(1.0, END)


'''
Edit functions for the GUI to apply to sequences
These include:
- Adding labels to number the sequences
- Adding where modifications occur
- Adding modification tags
- Removing all tags
'''


def gui_number_sequences(e):
    # For the lines in the text box
    counter = 1
    store_text = []
    new_text = []
    for line in my_text.get('1.0', 'end-1c').splitlines():
        # print('path: {}'.format(line))
        if line:
            store_text.append(line)
            text = '(staple' + "_" + str(counter) + ")" + store_text[counter - 1]
            new_text.append(text + '\n')
            # Add count
            counter += 1
    my_text.delete('1.0', 'end-1c')
    for line in range(len(new_text)):
        my_text.insert(str(float(line + 1)), new_text[line])


def gui_number_front_of_tags(e):
    # For the lines in the text box
    counter = 1
    store_text = []
    new_text = []
    for line in my_text.get('1.0', 'end-1c').splitlines():
        inserted_modification_tag_markers = re.findall(r'\/[^*]*\/', line)
        check_modification_added = len(inserted_modification_tag_markers)
        if check_modification_added >= 1:
            store_text.append(line)
            text = '(staple' + "_" + str(counter) + ")" + store_text[counter - 1]
            new_text.append(text + '\n')
            # Add count
            counter += 1
    my_text.delete('1.0', 'end-1c')
    for line in range(len(new_text)):
        my_text.insert(str(float(line + 1)), new_text[line])


def gui_mark_modification_start(e):
    position = my_text.index(tk.INSERT)
    my_text.insert(position, "**")


def gui_tag_modification(e):
    global selected
    if my_text.selection_get():
        # Grab selected text from text box
        selected = my_text.selection_get()
        user_prompt = 0
        if mb.askyesno('Add modification tag?', 'Do you want to add modification tag?'):
            user_prompt += 1
            # Add the tag modification
            selected = insert_modification_tag(selected, user_prompt)
            my_text.insert("sel.first", selected)
            my_text.delete("sel.first", "sel.last")
        else:
            pass


def gui_add_note(e):
    global selected
    if my_text.selection_get():
        # Grab selected text from text box
        selected = my_text.selection_get()
        if mb.askyesno('Add modification note?', 'Do you want to add a note about modification?'):
            # Add the tag modification
            selected = insert_modification_note_tag(selected)
            my_text.insert("sel.first", selected)
            my_text.delete("sel.first", "sel.last")
        else:
            pass


def gui_remove_tags(e):
    global selected
    if my_text.selection_get():
        # Grab selected text from text box
        selected = my_text.selection_get()
        if mb.askyesno('Delete selected tags?', 'Would you like to remove the selected tags?'):
            # Add the tag modification
            selected = return_raw_sequence(selected)
            my_text.insert("sel.first", selected[0])
            my_text.delete("sel.first", "sel.last")
        else:
            mb.showinfo('No', 'Removing Tags has been cancelled')


def gui_remove_tags_except_round(e):
    global selected
    if my_text.selection_get():
        # Grab selected text from text box
        selected = my_text.selection_get()
        if mb.askyesno('Delete selected tags?', 'Would you like to remove the selected tags?'):
            # Add the tag modification
            selected = return_raw_sequence_leave_round_brackets(selected)
            my_text.insert("sel.first", selected[0])
            my_text.delete("sel.first", "sel.last")
        else:
            mb.showinfo('No', 'Removing Tags has been cancelled')


def gui_seq_length(e):
    # For the lines in the text box
    counter = 1
    store_text = []
    new_text = []
    for line in my_text.get('1.0', 'end-1c').splitlines():
        # print('path: {}'.format(line))
        if line:
            new_line = return_raw_sequence(line)
            length_list = new_line[0]
            length = len(length_list)
            str_length = "(length: " + str(length) + ")"
            store_text.append(line)
            text = line + str_length
            new_text.append(text + '\n')
            # Add count
            counter += 1
    my_text.delete('1.0', 'end-1c')
    for line in range(len(new_text)):
        my_text.insert(str(float(line + 1)), new_text[line])


def bold():
    # create our font
    bold_font = font.Font(my_text, my_text.cget("font"))
    bold_font.configure(weight="bold")
    # configure tag
    my_text.tag_configure("bold", font=bold_font)
    # define tags
    current_tags = my_text.tag_names("sel.first")
    # if statement to see if tag is set
    if "bold" in current_tags:
        my_text.tag_remove("bold", "sel.first", "sel.last")
    else:
        my_text.tag_add("bold", "sel.first", "sel.last")


def selected(e):
    myLabel_1 = tk.Label(root, text=clicked.get())
    myLabel_1_text = clicked.get()
    global myLabel_1_global
    myLabel_1_global = myLabel_1_text


def selected_2(e):
    myLabel_2 = tk.Label(root, text=clicked_2.get())
    myLabel_2_text = clicked_2.get()
    global myLabel_2_global
    myLabel_2_global = myLabel_2_text


def fill_raw_panel_with_sequence(e):
    # remove the text prior
    if mb.askyesno('Verify', 'This creates raw sequences using left panel; continue?'):
        my_text_raw.delete('1.0', 'end-1c')
        # For the lines in the text box
        print("Attempting to fill raw sequences panel (right)")
        counter = 1
        store_text = []
        new_text = []
        for line in my_text.get('1.0', 'end-1c').splitlines():
            # print('path: {}'.format(line))
            if line:
                new_line = return_raw_sequence(line)
                length_list = new_line[0]
                length = len(length_list)
                store_text.append(line)
                text = new_line[0]
                new_text.append(text + '\n')
                # Add count
                counter += 1
        for line in range(len(new_text)):
            my_text_raw.insert(str(float(line + 1)), new_text[line])
        return


def fill_raw_panel_with_sequence_button():
    # remove the text prior
    if mb.askyesno('Verify', 'This uses left panel data to replace and modify right panel; continue?'):
        my_text_raw.delete('1.0', 'end-1c')
        # For the lines in the text box
        print("Attempting to fill raw sequences panel (right)")
        counter = 1
        store_text = []
        new_text = []
        for line in my_text.get('1.0', 'end-1c').splitlines():
            # print('path: {}'.format(line))
            if line:
                new_line = return_raw_sequence(line)
                length_list = new_line[0]
                length = len(length_list)
                store_text.append(line)
                text = new_line[0]
                new_text.append(text + '\n')
                # Add count
                counter += 1
        for line in range(len(new_text)):
            my_text_raw.insert(str(float(line + 1)), new_text[line])
        return


def colour_round_brackets(marked_sequence):
    # origami specific tag
    match_origami_specific_tag = re.sub(r'\([^)]*\)', '', marked_sequence)
    return match_origami_specific_tag


def colour_square_brackets(marked_sequence):
    # modification name tag
    match_modification_name_tag = re.sub(r'\[[^]]*\]', '', marked_sequence)
    return match_modification_name_tag


def colour_forward_slash(marked_sequence):
    # modification tag
    match_modification_tag = re.sub(r'\/[^)]*\/', '', marked_sequence)
    return match_modification_tag


def fill_raw_panel_with_highlights(e):
    # # remove the text prior
    # my_text_raw.delete('1.0', 'end-1c')
    # # For the lines in the text box
    # counter = 1
    # store_text = []
    # new_text = []
    # for line in my_text.get('1.0', 'end-1c').splitlines():
    #     # print('path: {}'.format(line))
    #     if line:
    #         store_text.append(line)
    #         text = line
    #         new_text.append(text + '\n')
    #         # Add count
    #         counter += 1
    # for line in range(len(new_text)):
    #     my_text_raw.insert(str(float(line + 1)), new_text[line])
    # # Here you add the render changes such as bold
    # bold_font = font.Font(my_text_raw, my_text_raw.cget("font"))
    # bold_font.configure(weight="bold")
    # # configure tag
    # my_text.tag_configure("bold", font=bold_font)
    # # define tags
    # current_tags = my_text.tag_names("sel.first")
    # # if statement to see if tag is set
    # if "bold" in current_tags:
    #     my_text.tag_remove("bold", "sel.first", "sel.last")
    # else:
    #     my_text.tag_add("bold", "sel.first", "sel.last")
    return


def save_as_raw_seq_file():
    # save raw sequences as file
    text_file = filedialog.asksaveasfilename(defaultextension=".*", initialdir="C:/", title="Save File",
                                             filetypes=(("Text Files", "*.txt"),
                                                        ("CSV Files", "*.csv"),
                                                        ("All FileTypes", "*.*")))
    if text_file:
        # update the status in GUI
        name = text_file
        status_bar.config(text=f'{name}')
        name = name.replace("C:/", "")
        root.title(f'{name} - Sequence Editor')

        # save the file
        text_file = open(text_file, 'w')
        text_file.write(my_text_raw.get(1.0, END))
        # close the file
        text_file.close()


def save_raw_seq_file():
    global open_file_name_global
    if open_file_name_global:
        # save the file
        text_file = open(open_file_name_global, 'w')
        text_file.write(my_text_raw.get(1.0, END))
        # close the file
        text_file.close()
        # pop-up box code
        status_bar.config(text=f'Saved: {open_file_name_global}')
    else:
        save_as_file()


def clear_right_panel(e):
    if mb.askyesno('Verify', 'Clear right panel? This will remove raw sequences'):
        # delete the raw / right panel sequences
        my_text_raw.delete('1.0', END)
        return


def clear_left_panel(e):
    if mb.askyesno('Verify', 'Clear left panel? This will remove left panel sequences'):
        # delete the left panel sequences
        my_text.delete('1.0', END)
        return


def clear_both_panels(e):
    if mb.askyesno('Verify', 'Clear both left / right panels? This will remove all sequences'):
        # delete the raw / right panel sequences
        my_text_raw.delete('1.0', END)
        # delete the left panel sequences
        my_text.delete('1.0', END)
        return
    return


def create_markdown_render():
    print("Attempting to create markdown")
    if mb.askyesno('Verify', 'Create markdown HTML document of Left Panel Sequences?'):
        # has to send text (left panel) to the renderer
        my_text_to_render = my_text.get("1.0", END)
        # add header text
        input_text = add_header(my_text_to_render)
        # replace marked-up with markdown pipes and hyphen headers
        table_formatted_text = create_markdown_table(input_text)
        # fill in raw sequence blanks
        markdown_table_pre_parsed = fill_in_raw_seq_fields(table_formatted_text)
        # use mistune parser to form html format and create a saved file
        parsed = parse_markdown(markdown_table_pre_parsed)

        # create pre-defined HTML string for start of file
        pre_header_str = """
        <!DOCTYPE html>
        <html>
        <link rel="stylesheet" type="text/css" href="origami_sequence_style_sheet.css" />
        """

        # add this to the top of parsed multi-line string
        parsed_pre_defined_html = pre_header_str + parsed

        # store the content onto that saved file
        html_file = filedialog.asksaveasfilename(defaultextension=".*", initialdir="C:/",
                                                 title="Render HTML File",
                                                 filetypes=(("HTML File", "*.html"),
                                                            ("All FileTypes", "*.*")))
        if html_file:
            # save the file
            html_str_to_store = parsed_pre_defined_html
            # create pre-defined HTML file
            html_file = open(html_file, "w")
            # actual table content appended to written html file
            html_file.write(html_str_to_store)
            html_file.close()
    return


# pop up for read-me
def pop_readme():
    global pop
    pop = tk.Toplevel(root)
    pop.title("Read me")
    pop.geometry("500x800")
    markdown_readme_text = """

    ////////////////////////////////////////////////////////////
        This is the readme text: 
        Use this tool to import DNA / RNA Origami staple and scaffold sequences
        and "mark-up" the modifications for downstream tasks 
    ////////////////////////////////////////////////////////////


    /////////////////////////// bindings ///////////////////////

        '<Control-A>', select_all                            
        '<Control-Z>', undo                                  
        '<Control-Y>', redo                                  
        '<Control-X>', gui_remove_tags                       
        
        
        
    /////////////////// staple modification bindings ///////////

        '<Control-Q>', gui_mark_modification_start           
        '<Control-W>', gui_tag_modification                  
        '<Control-E>', gui_add_note        
        
        
    /////////////////// step by step guide /////////////////////        
    
        IMPORT / OPEN SEQUENCES
    1) Import staple sequences as text file using File Commands - Open Sequence File
    2) Sequences should appear to be marked up in the left panel
    
        MARK SEQUENCE MODIFICATION LOCATION 
    3) Use Control Q to "mark" using asterixes where sequences are modified
    
        SELECT MODIFICATION NOTE AND TYPE 
    4) Select using the buttons in the top left what the modifications are
    
        HIGHLIGHT AND ADD MODIFICATION NOTE AND TYPE
    5) Press Control-A to highlight the sequences in the left panel
    6) Press Control-W to tag the modification. Check this is correct.
    7) Press Control-A to highlight the sequences again.
    8) Press Control-E to add a modification note.
    9) Check the sequences are correct.
    
        HIGHLIGHT AND ADD STAPLE NUMBER AND LENGTH
    10) Press Control-A and to highlight the sequences one last time.
    11) Open the Edit Sequences panel to add number and lengths of sequence.
    
        VIEW RAW SEQUENCES AND RENDER
    12) Press the "Show Raw Sequences" to obtain the raw sequences.
    13) Press the "Render Mark-up HTML" to view HTML table of sequences
     
    ////////////////////////////////////////////////////////////
    """
    pop_label = tk.Label(pop, text=markdown_readme_text)
    pop_label.pack(pady=10)


def undo(self, event=None):
    ...


def redo(self, event=None):
    ...


clicked = tk.StringVar()
clicked.set("modification use")
uses_drop = tk.OptionMenu(root, clicked, *list_of_modifications, command=selected)
uses_drop.pack(anchor=tk.W)

clicked_2 = tk.StringVar()
clicked_2.set("specific modification name")
names_drop = tk.OptionMenu(root, clicked_2, *list_of_names, command=selected_2)
names_drop.pack(anchor=tk.W)

# create main frame
my_frame = tk.Frame(root)

# create toolbar
toolbar_frame = tk.Frame(my_frame)
toolbar_frame.pack(anchor=tk.CENTER)

# create scrollbar
vertical_scroll = tk.Scrollbar(my_frame)
vertical_scroll.pack(side=tk.RIGHT, fill=tk.Y)

# create horizontal scrollbar
horizontal_scroll = tk.Scrollbar(my_frame, orient='horizontal')
horizontal_scroll.pack(side=tk.BOTTOM, fill=tk.X)

# create text box
my_text = tk.Text(my_frame, width=76, height=40, font=("Arial", 16), undo=True,
                  yscrollcommand=vertical_scroll.set, xscrollcommand=horizontal_scroll, wrap="none")
# my_text.tag_configure(tagName="red", foreground="#ff0000")
# my_text.highlight_pattern(pattern='\*[^*]*\*', tag="red")
my_text.pack(side=tk.LEFT)

# create raw text box
my_text_raw = tk.Text(my_frame, width=76, height=40, font=("Arial", 16), selectbackground="green",
                      selectforeground="black", undo=True, wrap="none")
my_text_raw.pack(side=tk.RIGHT)
my_frame.pack(ipady=8)

# config scrollbar
vertical_scroll.config(command=my_text.yview)
horizontal_scroll.config(command=my_text.xview)

# create menu
my_menu = tk.Menu(root)
root.config(menu=my_menu)

# add file menu
file_menu = tk.Menu(my_menu, tearoff=False)
my_menu.add_cascade(label="File Commands", menu=file_menu)
file_menu.add_command(label="New File", command=new_file)
file_menu.add_command(label="Open Sequence File", command=open_file)
file_menu.add_command(label="Save File", command=save_file)
file_menu.add_command(label="Save File As", command=save_as_file)
file_menu.add_separator()
file_menu.add_command(label="Read-me / Help", command=pop_readme)
file_menu.add_separator()
file_menu.add_command(label="Exit", command=callback)

edit_menu = tk.Menu(my_menu, tearoff=False)
my_menu.add_cascade(label="Edit Sequences", menu=edit_menu)
edit_menu.add_command(label="Add numbered tags for highlighted sequences",
                      command=lambda: gui_number_sequences(False))
edit_menu.add_command(label="Add sequence length tags at end of highlighted sequences",
                      command=lambda: gui_seq_length(False))
edit_menu.add_separator()
edit_menu.add_command(label="Clear all selected tags: [Ctrl+X]", command=lambda: gui_remove_tags(False))
edit_menu.add_command(label="Clear all selected tags except round brackets",
                      command=lambda: gui_remove_tags_except_round(False))

modification_menu = tk.Menu(my_menu, tearoff=False)
my_menu.add_cascade(label="Modify Sequences", menu=modification_menu)
modification_menu.add_command(label="Step 1: Mark where sequence has been modified: [Ctrl+Q]",
                              command=lambda: gui_mark_modification_start(False))
modification_menu.add_command(label="Step 2: Add tag for modification chosen: [Ctrl+W]",
                              command=lambda: gui_tag_modification(False))
modification_menu.add_command(label="Step 3: Add modification note: [Ctrl+E]", command=lambda: gui_add_note(False))

raw_sequence_menu = tk.Menu(my_menu, tearoff=False)
my_menu.add_cascade(label="Left/Right Panel Options", menu=raw_sequence_menu)
raw_sequence_menu.add_command(label="Show raw sequences in panel",
                              command=lambda: fill_raw_panel_with_sequence(False))
raw_sequence_menu.add_command(label="Create rendered text", command=lambda: create_markdown_render())
raw_sequence_menu.add_separator()
raw_sequence_menu.add_command(label="Save Right Panel file",
                              command=lambda: save_raw_seq_file)
raw_sequence_menu.add_command(label="Save Right Panel file as",
                              command=lambda: save_as_raw_seq_file)
raw_sequence_menu.add_separator()
raw_sequence_menu.add_command(label="Clear (Remove) Left Panel Sequences",
                              command=lambda: clear_left_panel(False))
raw_sequence_menu.add_command(label="Clear (Remove) Right Panel Sequences",
                              command=lambda: clear_right_panel(False))
raw_sequence_menu.add_command(label="Clear (Remove) All Sequences",
                              command=lambda: clear_both_panels(False))

# add bindings to the editor
root.bind('<Control-A>', select_all)
root.bind('<Control-a>', select_all)
root.bind('<Control-Z>', undo)
root.bind('<Control-z>', undo)
root.bind('<Control-Y>', redo)
root.bind('<Control-y>', redo)
root.bind('<Control-X>', gui_remove_tags)
root.bind('<Control-x>', gui_remove_tags)
# add modification bindings
root.bind('<Control-Q>', gui_mark_modification_start)
root.bind('<Control-q>', gui_mark_modification_start)
root.bind('<Control-W>', gui_tag_modification)
root.bind('<Control-w>', gui_tag_modification)
root.bind('<Control-E>', gui_add_note)
root.bind('<Control-e>', gui_add_note)

# add status bar to bottom
status_bar = tk.Label(root, text="Ready        ", anchor=tk.E)
status_bar.pack(fill=tk.X, side=tk.BOTTOM, ipady=10)

# buttons
raw_seq_button = tk.Button(toolbar_frame, text='Show Raw Sequences',
                           command=fill_raw_panel_with_sequence_button)
raw_seq_button.pack()

markdown_render_button = tk.Button(toolbar_frame, text='Render Mark-Up HTML',
                                   command=create_markdown_render)
markdown_render_button.pack()

render_menu = tk.Menu()
tk.mainloop()
