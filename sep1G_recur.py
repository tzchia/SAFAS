import os, glob, zipfile

def zipdir(path, ziph):
    _size = 0
    # ziph is zipfile handle
    '''
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))
            _size += os.path.getsize(os.path.join(root, file))
    '''
    ziph.write(path)
    _size += os.path.getsize(path)
    return _size/1024./1024.

fs = glob.glob('Oulu_train/*') #TODO
fs.sort()
i, currSize = 0, 0
zipf = zipfile.ZipFile('1_1_01_1.avi.zip', 'w', zipfile.ZIP_DEFLATED)
while i < len(fs):
    if currSize < 950:
        currSize += zipdir(fs[i], zipf)
        i += 1
    else:
        zipf.close()
        print(currSize)
        currSize = 0
        zipf = zipfile.ZipFile(f'{fs[i]}.zip', 'w', zipfile.ZIP_DEFLATED)
zipf.close()
