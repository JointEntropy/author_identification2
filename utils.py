import pickle
import os
import numpy as np
import json
from scipy.sparse import csr_matrix


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)


class DriveWrapper:
    def __init__(self):
        from pydrive.auth import GoogleAuth
        from pydrive.drive import GoogleDrive
        from google.colab import auth
        from oauth2client.client import GoogleCredentials

        # Authenticate and create the PyDrive client.
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)

    def print_rootdir(self):
        for file1 in self.drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList():
            print('title: %s, id: %s' % (file1['title'], file1['id']))

    def __search_by_name(self, name):
        for file_ in self.drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList():
            if file_['title']==name:
                return file_['id']
        raise ValueError('File with {} does not exist.'.format(name))

    def download(self, name):
        id_ = self.__search_by_name(name)
        print(id_)
        myfile = self.drive.CreateFile({'id': id_})
        myfile.GetContentFile(myfile['title'])  # Save Drive file as a local file
        print('"{}" saved!'.format(myfile['title']))
        return myfile['title']

    def save(self, filename, save_name=None):
        name = save_name or filename
        uploaded = self.drive.CreateFile({'title': name})
        uploaded.SetContentFile(filename)
        uploaded.Upload()
        print('File "{}" uploaded with name "{}" and with ID {}'.format(filename, name, uploaded.get('id')))


def json_load(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)


def save_sparse(matr, pth):
    with open(pth, 'wb') as f:
        pickle.dump(csr_matrix(matr), f)


def split_sequence(seq, maxlen):
    """
    Split text into parts of size inputlen.
    :param seq:
    :param maxlen: max len(in tokens) of text.
    :return:
    """
    seqs = np.array_split(list(seq), int(len(seq) / maxlen))
    return seqs


if __name__ == '__main__':
    gdw = DriveWrapper()
    gdw.list_root()
    # gdw.download(*id*)


