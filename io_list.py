import sys

def save_list(list_obj, path):
    with open(path, 'w') as filehandle:
        for listitem in list_obj:
            filehandle.write('%s\n' % listitem)

def load_list(path):
    # define an empty list
    list_obj = []

    # open file and read the content in a list
    with open(path, 'r') as filehandle:
        for line in filehandle:
            # remove linebreak which is the last character of the string
            listitem = line[:-1]

            # add item to the list
            list_obj.append(listitem)

    return list_obj