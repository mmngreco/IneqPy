if __name__ == '__main__':
    import os
    import sys
    import shutil
    folders = ['_build', '_autodoc', '_autosummary']
    [shutil.rmtree(folder) for folder in folders if os.path.exists(folder)]
    os.system(r'sphinx-apidoc -M -T -f -e -o _autodoc ../ineqpy')
    os.system(r'sphinx-build -b html . _build/html')
