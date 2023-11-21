import os

# define the name of the directory to be created
path = "/manual_paper_designs_schematics/"
print(path)

try:
    for file_number in range(1, 124):
        name = "Paper_" + str(file_number) + "_Design_Images"
        print(name)
        os.makedirs(path + name)

except OSError:
    print("Creation of the directory %s failed" % path)
else:
    print("Successfully created the directory %s" % path)
