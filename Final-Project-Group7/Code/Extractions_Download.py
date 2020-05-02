import pytator
import argparse
import pickle
import os
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--url", required=True)
parser.add_argument("--project", required=True)
parser.add_argument("--token", required=True)

# The project number for NOAA OER data is 20
# The token is an example token, you can substitute your own by visiting https://www.tatorapp.com/rest/#Token-create, and entering your credentials

argstring = ['--url', 'https://www.tatorapp.com/rest/', '--project', '20', '--token', '9d8ea1721ecaee17e6462316e73cea03732fcfa1']

args = parser.parse_args(argstring)

tator = pytator.Tator(args.url.rstrip('/'), args.token, args.project)
image_medias = tator.Media.filter({"type" : 81}) # Filters to images only

FILE_LIST="path/to/labels.csv"
filtered_medias = []
fnames = pd.read_csv(FILE_LIST, header=None, names=['filename','label'])
for media in image_medias:
    if len(fnames[fnames['filename']==media.get('name')])==1:
        filtered_medias.append(media)

# You will need to specify a download directory here, make note of the escaped backslashes if in Windows
DOWNLOAD_DIRECTORY = 'C:\\Users\\bgwoo\\Documents\\Projects\\NOAA OER'
# This file will let you resume or retry downloads that don't finish or fail
SUCCEED_FILE = os.path.join(DOWNLOAD_DIRECTORY, 'success.pkl')

if os.path.isfile(SUCCEED_FILE):
    filter_list = pickle.load(open(SUCCEED_FILE,'rb'))
else:
    filter_list = []

for media in tqdm(filtered_medias):
    if media.get('name') in filter_list:
        continue
    # Downloads media file to a local file with the same name
    # as the database entry
    try:
        tator.Media.downloadFile(media, os.path.join(DOWNLOAD_DIRECTORY,media.get('name')))
        filter_list.append(media.get('name'))
        with open(SUCCEED_FILE, 'wb') as f:
            pickle.dump(filter_list,f)
    except:
        print(media.get('name') + " not downloaded")