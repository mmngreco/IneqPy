if __name__ == '__main__':

    import os
    import sys
    import shutil

    to_remove_list = ['_build', '_autodoc', '_autosummary', '_autodoc',
                      '_autosummary', '_modules', '_sources', '_static',
                      'api.html', 'genindex.html', 'index.html', 'objects.inv',
                      'py-modindex.html', 'search.html', 'searchindex.js',
                      ]

    for folder_file in to_remove_list:
        if not os.path.exists(folder_file): continue
        try:
            shutil.rmtree(folder_file)
        except:
            os.remove(folder_file)

    os.system(r'sphinx-apidoc -M -T -f -e -o _autodoc ../ineqpy')
    os.system(r'sphinx-build -b html ./source/ .')
