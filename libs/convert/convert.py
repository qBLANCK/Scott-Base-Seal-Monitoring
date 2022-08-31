import subprocess
from os import listdir, makedirs
from os.path import join, isfile, isdir, basename
from pathlib import Path

from tqdm import tqdm

'''
Recursivley search through an input directory (in_dir) and convert all .rw2 files into .jpg.
Skipping previously converted images (in-case of script failure before completion).
Keeps the directory structure and image names.
'''

cur_pth = Path().resolve()
dcraw = cur_pth / 'dcraw/usr/bin/dcraw'
out_file_ext = 'jpg'
in_dir = cur_pth / 'in/'
in_files = [f for f in listdir(in_dir) if isfile(join(in_dir, f))]
out_dir = cur_pth / 'out'


def convert_image(file_path, dir):
    img_name = basename(file_path).split('.')[0]
    process_1 = subprocess.Popen([dcraw, '-w', '-c', in_dir / file_path], cwd=dir, stdout=subprocess.PIPE)
    process_2 = subprocess.Popen(['convert', '-', f'{img_name}.{out_file_ext}'], cwd=dir, stdin=process_1.stdout,
                                 stdout=subprocess.PIPE)

    out, error = process_2.communicate()
    return out, error


def recursion(path=Path('./')):
    items = listdir(in_dir / path)
    print(path)
    for item in tqdm(items):
        if isfile(in_dir / path / item):
            file_name, file_ext = basename(in_dir / path / item).split('.')  # Assumption: Only one '.' in file name
            if file_ext.lower() == 'rw2' and not isfile(out_dir / path / f'{file_name}.{out_file_ext}'):
                convert_image(path / item, out_dir / path)
            else:
                print(f'Skipping file "out/{path / item}"')

        elif isdir(in_dir / path / item):
            makedirs(out_dir / path / item, exist_ok=True)  # Won't overwrite existing dir or children
            recursion(path / item)


recursion()
