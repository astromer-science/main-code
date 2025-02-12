import os
import pandas as pd

from presentation.pipelines.machofields.utils import connect_to_drive

PATH = './data/temp' # temporal directory to store data
os.makedirs(PATH, exist_ok=True)
drive = connect_to_drive()
file_list = drive.ListFile({'includeItemsFromAllDrives':True,
                            'driveId':'0ANj7Lpiu7QacUk9PVA',
                            'corpora':'drive',
                            'supportsAllDrives':True,
                            'q': "'1HaRYj6lA2uIS4poQw2yIZYZaRTmD82Wu' in parents and trashed=false"}).GetList()

ids = pd.DataFrame([{'name': file['title'], 'id': file['id']} for file in file_list])
ids.to_csv(os.path.join(PATH, 'ids.csv'))


