

import os
import ssl
from http.client import IncompleteRead
from numbers import Number
from pathlib import Path
from sys import platform
import urllib

import numpy as np
import progressbar
import requests
import shutil

class Input_output_data:
    def __init__(self, u, y, sampling_time=None, name=None, state_initialization_window_length=None):
        assert len(u)==len(y), f'input sequence u need to have the same length as y: currently {u.shape=}, {y.shape=}'
        self.u = u
        self.y = y
        self.sampling_time = sampling_time
        self.name = '' if name is None else name
        self.state_initialization_window_length = state_initialization_window_length
    
    def __repr__(self):
        z = '' if (self.name==None or self.name=='') else f' "{self.name}"' 
        u, y = self.u, self.y
        A = f'sampling_time={float(self.sampling_time):.4}' if isinstance(self.sampling_time, Number) else 'sampling_time=Discrete time'
        Z = f' state_initialization_window_length={self.state_initialization_window_length}' if self.state_initialization_window_length!=None else ''
        return f'Input_output_data{z} {u.shape=} {y.shape=} {A}{Z}'
    
    def __iter__(self):
        yield self.u
        yield self.y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self,arg):
        '''Slice the System_data in time index'''
        if isinstance(arg, int):
            if arg==0:
                return self.u
            elif arg==1:
                return self.y
            raise ValueError(f'if argument {arg} is a int than only 0 for u and 1 for y can be used')
        elif isinstance(arg,slice):
            start, stop, step = arg.indices(len(self.u))
            unew = self.u[arg]
            ynew = self.y[arg]
            return Input_output_data(u=unew, y=ynew, sampling_time=self.sampling_time)
        else:
            raise ValueError(f'argument with value "{arg}" of __getitem__ of type "{type(arg)}" is not allowed')
    
    def atleast_2d(self):
        v = lambda x: x if x.ndim>1 else x[:,None]
        return Input_output_data(u=v(self.u), y=v(self.y), \
                                sampling_time=self.sampling_time, \
                                name=self.name, \
                                state_initialization_window_length=self.state_initialization_window_length)

    def split(self, frac):
        s = int(len(self)*(1-frac))
        return self[:s], self[s:]

def atleast_2d_fun(*data, apply=True):
    if len(data)==1:
        data = data[0]
    if apply==False:
        return data
    if isinstance(data, Input_output_data):
        return data.atleast_2d()
    else:
        return tuple(atleast_2d_fun(d, apply=apply) for d in data)


def always_return_tuples_of_datasets_fun(train, test, apply=False):
    if apply==False:
        return train, test
    train = (train,) if not isinstance(train, (list,tuple)) else train
    test = (test,) if not isinstance(test, (list,tuple)) else test
    return train, test


def get_tmp_benchmark_directory():
    '''A utility function which gets the utility directories for each OS

    It creates a working directory called nonlinear_benchmarks 

        in LOCALAPPDATA for windows

        in ~/.nonlinear_benchmarks/ for unix like

        in ~/Library/Application Support/nonlinear_benchmarks/ for darwin

    it creates two directories inside of the nonlinear_benchmarks directory

        data_sets : cache location of the downloaded data sets

        checkpoints : used during training of torch models

    Returns
    -------
    dict(base=base_dir, data_sets=data_sets_dir, checkpoints=checkpoints_dir)
    '''

    def mkdir(directory):
        if os.path.isdir(directory) is False:
            os.mkdir(directory)

    from sys import platform
    if platform == "darwin": #not tested but here it goes
        base_dir = os.path.expanduser('~/Library/Application Support/nonlinear_benchmarks/')
    elif platform == "win32":
        base_dir = os.path.join(os.getenv('LOCALAPPDATA'),'nonlinear_benchmarks/')
    else: #unix like, might be problematic for some weird operating systems.
        base_dir = os.path.expanduser('~/.nonlinear_benchmarks/')#Path('~/.nonlinear_benchmarks/')
    mkdir(base_dir)
    return base_dir

def clear_cache():
    '''Delete all cached downloads'''
    
    temp_dir = get_tmp_benchmark_directory()
    for l in ['EMPS','CED','F16','WienHammer','BoucWen','ParWHF','WienerHammerBenchMark','Silverbox','Cascaded_Tanks']:
        try:
            shutil.rmtree(os.path.join(temp_dir,l))
        except FileNotFoundError:
            pass
    try:
        shutil.rmtree(os.path.join(temp_dir,'DaISy_data'))
    except FileNotFoundError:
        pass


class MyProgressBar():
    def __init__(self,download_size):
        self.pbar = None
        self.download_size = download_size

    def __call__(self, block_num, block_size, total_size):
        total_size = self.download_size
        if not self.pbar:
            self.pbar=progressbar.ProgressBar(maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

def cashed_download(url,name_dir,zip_name=None,dir_placement=None,download_size=None,force_download=False,zipped=True):
    '''url is the file to be downloaded
    name_dir is the directory name where the file and the contents of the file will be saved
    dir_placement is an optional argument that gives the location of the downloaded file 
    if it is none it will download to the temp dir
    if dir_name is None it will be saved in the temp directory of the system'''

    #finding/making directories
    if isinstance(url, list):
        urls = url
        for url in urls:
            try:
                return cashed_download(url, name_dir, zip_name=zip_name,\
                                        dir_placement=dir_placement,download_size=download_size,force_download=force_download, zipped=zipped)
            except Exception as e:
                print(f'Unable to download dataset with {url=} since a "{e}" exception was raised')
        raise ValueError(f'unable to download with urls: {urls}')

    if dir_placement is None:
        p = get_tmp_benchmark_directory() #use temp dir
    else:
        p = Path(dir_placement) #use given dir
    save_dir = os.path.join(p,Path(name_dir))
    if os.path.isdir(save_dir) is False:
        os.mkdir(save_dir)
    file_name = url.split('/')[-1] if zip_name==None else zip_name
    save_loc = os.path.join(save_dir,file_name) 

    if os.path.isfile(save_loc) and not force_download:
        return save_dir

    if force_download:
        print(f'(re-)downloading dataset from {url} \n in {save_loc}')
    else:
        print(f'dataset not found downloading from {url} \n in {save_loc}')

    if 'drive.google' in url:
        download_file_from_google_drive(url, save_loc)
    else:
        # Temporarily disable SSL verification for this download
        original_context = ssl._create_default_https_context
        
        tries = 0
        while True:
            try:
                # Monkey patch SSL verification
                ssl._create_default_https_context = ssl._create_unverified_context
                
                if download_size is None:
                    urllib.request.urlretrieve(url, save_loc)
                    break
                else:
                    urllib.request.urlretrieve(url, save_loc, MyProgressBar(download_size=int(download_size)))
                    break
                    
            except (IncompleteRead, urllib.error.URLError) as e:
                tries += 1
                print(f'Download attempt {tries} failed: {e}')
                if isinstance(e, IncompleteRead):
                    print('IncompleteRead download failed, re-downloading file')
                    download_size = None
                if tries == 5:
                    raise Exception(f'Download failed after 5 attempts: {e}')
            finally:
                # Always restore original SSL context
                ssl._create_default_https_context = original_context

    if not zipped: 
        return save_dir
    
    print('extracting file...')
    print(f'{save_loc=}')
    ending = file_name.split('.')[-1]
    if ending=='gz':
        import shutil
        import gzip
        with open(os.path.join(save_dir,file_name[:-3]), 'wb') as f_out:
            with gzip.open(save_loc, 'rb') as f_in:
                shutil.copyfileobj(f_in, f_out)
    elif ending=='zip':
        from zipfile import ZipFile as File
        with File(save_loc) as Obj:
            Obj.extractall(save_dir)
    elif ending=='rar':
        from rarfile import RarFile as File
        with File(save_loc) as Obj:
            Obj.extractall(save_dir)
    else:
        raise NotImplementedError(f'file {file_name} type not implemented')
    return save_dir

def download_file_from_google_drive(url, destination):
    id = url.split('/')[-2]
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

# if __name__ == '__main__':
#     import nonlinear_benchmarks
#     u,y = nonlinear_benchmarks.CED(split_data=False)
#     sys_data.plot(show=True)

    # filename = './file.zip'
    # url = 'http://www.nonlinearbenchmark.org/FILES/BENCHMARKS/EMPS/EMPS.zip'
    # urllib.request.urlretrieve(url, filename)

    # resp = urllib.request.urlopen(url)
    # respHtml = resp.read()
    # binfile = open(filename, "wb")
    # binfile.write(respHtml)
    # binfile.close()

    