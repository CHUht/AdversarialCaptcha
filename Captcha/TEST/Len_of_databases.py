import os

path = os.getcwd()
path = path.replace("\TEST","")

path = path + "/FILTER_ADVERSARIAL_DATABASE/"

listing = os.listdir(path)
print(len(listing))