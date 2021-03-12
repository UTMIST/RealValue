
import os
import shutil
# iterate thorugh image file names so that it starts from 1

source = 'downloaded_toronto_raw_dataset'
dataset_dir = 'toronto_raw_dataset'
offset = 133 # Number of houses that "first house" in dataset is away from 1
#new_dir_path = 'toronto_raw_dataset'
#if os.path.isdir(new_dir_path) == False:
#    os.mkdir(new_dir_path)

if os.path.isdir(dataset_dir) == True:
    shutil.rmtree(dataset_dir)
shutil.copytree(source, dataset_dir)

#files = os.scandir(dataset_dir)
'''
for entry in files:
    print("Filename", entry.path)
    if entry.is_file() and (not entry.path.endswith(".txt")):
        entry_string = os.path.basename(entry) #just the filename, not the full path

        # filename has format <NUM>_<VIEW>.jpeg
        splitted = entry_string.split("_") #e.g. this is ['134', 'frontal.jpg']
        print("Splitted", splitted)
        number = int(splitted[0])
        #print("Old number", number)
        new_number = number - offset
        #print("New number", new_number)

        new_filename = os.path.join(dataset_dir, str(new_number) + '_' + splitted[1])

        os.rename(entry.path, new_filename)
'''
def list_dir_function():
    files = os.listdir(dataset_dir)
    print(files)
    for filename in files:
        if not filename.endswith(".txt"):
            splitted = filename.split("_") #e.g. this is ['134', 'frontal.jpg']
            print("Splitted", splitted)
            number = int(splitted[0])
            #print("Old number", number)
            new_number = number - offset
            # print(os.path.join(directory, filename))
            new_filename = os.path.join(dataset_dir, str(new_number) + '_' + splitted[1])

            os.rename(os.path.join(dataset_dir, filename), new_filename)

list_dir_function()
