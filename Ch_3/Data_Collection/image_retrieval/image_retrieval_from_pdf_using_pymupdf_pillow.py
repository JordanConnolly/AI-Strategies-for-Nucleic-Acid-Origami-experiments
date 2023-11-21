# python -m pip install PyMuPDF Pillow

# for image extraction
import io
import fitz
from PIL import Image

# for file path
import glob
import re
import time


"""
    Extract images from PDF files automatically. Create database of images
    The workflow should be as follows:
    access the pdf file (full text pdf; supplementary pdf / supplementary pdf 1), store name in list
    extract the images from the specified pdf file, to drive and store url if available
    save lists and create a count of images per paper
    save image as 'pdf_file_name' and '_count' 
    create a data frame from the url / drive image name / pdf name / count
    turn the data frame into a csv to be uploaded into a SQL database
"""


def numerical_sort(value):
    numbers = re.compile(r'(\d+)')
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts


# file path you want to extract images from
# file_path_of_main_paper_pdf = "D:/Database_Back_Ups_RQ1/DataBase_07_05_2021_additions/Main_PDF_Files/*.pdf"
file_path_of_suppl_paper_pdf = "D:/Database_Back_Ups_RQ1/DataBase_07_05_2021_additions/Suppl_PDF_Files/*.pdf"
# file = "pdf_main_101.pdf"
# all_pdf_main_files = sorted(glob.glob(file_path_of_main_paper_pdf), key=numerical_sort)
all_pdf_suppl_files = sorted(glob.glob(file_path_of_suppl_paper_pdf), key=numerical_sort)

storage_path = ""  # make sure the storage path is somewhere that can take at least 2GB storage.
# I am using an external hard drive with at least 1TB of storage space. An example 15 papers stored 12mb.


for file in all_pdf_suppl_files:
    file_name = file.split("\\")[-1].split(".")[0]
    print(file_name)
    # open the file
    pdf_file = fitz.open(file)
    # iterate over pdf pages
    for page_index in range(len(pdf_file)):
        # get the page itself
        page = pdf_file[page_index]
        image_list = page.get_images()
        # printing number of images found in this page
        if image_list:
            print(f"[+] Found a total of {len(image_list)} images in page {page_index}")
        else:
            print("[!] No images found on page", page_index)
        for image_index, img in enumerate(page.get_images(), start=1):
            # get the XREF of the image
            xref = img[0]
            # extract the image bytes
            base_image = pdf_file.extract_image(xref)
            try:
                image_bytes = base_image["image"]
                # get the image extension
                image_ext = base_image["ext"]
                # load it to PIL
                image = Image.open(io.BytesIO(image_bytes))
                print("extension:", image_ext)
                # save it to local disk
                image.save(open(f"all_photos_from_suppl_paper_pdf/"
                                f"{file_name}_page_{page_index + 1}_image{image_index}.{image_ext}", "wb"))
            except:
                image_ext = base_image["ext"]
                print(image_ext + " as extension didn't work")
