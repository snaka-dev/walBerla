import argparse

#!/usr/bin/env python3

import os

def cmake_template ( file_list, module_name, top_level ):
    newline = '\n    '
    if top_level:
        return f"""add_library( {module_name} )
target_sources( {module_name}
    PRIVATE
    {newline.join(x for x in file_list)}     
    )  
"""
    else:
        return f"""target_sources( {module_name}
    PRIVATE
    {newline.join(x for x in file_list)}     
    )  
"""



def create_cmake_lists (folder, module_name, top_level=True):
    header_and_source = []
    newline = '\n    '
    for entry in os.scandir(folder):
        if entry.name == 'all.h':
            continue
        if entry.is_file():
            if entry.name.endswith('.h') or entry.name.endswith('.cpp'):
                header_and_source += [entry.name]
        else:
            create_cmake_lists(folder + '/' + entry.name, module_name, False)

    if not header_and_source:
        return

    if top_level:
        with open(folder + '/CMakeListsRefactor.txt', 'w') as f:
            content = f"""add_library( {module_name} )
target_sources( {module_name}
    PRIVATE
    {newline.join(x for x in header_and_source)}     
    )
"""
            f.write(content)
    else:
        with open(folder + '/CMakeLists.txt', 'w') as f:
            content = f"""target_sources( {module_name}
    PRIVATE
    {newline.join(x for x in header_and_source)}     
    )
"""
            f.write(content)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refactor CMakeLists.txt file')
    parser.add_argument('folder', type=str, help='Folder to be refactored; a CMakeListsRefactor.txt file will be created in each subfolder')
    args = parser.parse_args()

    create_cmake_lists( args.folder, args.folder.split('/')[-1] )
    # print(args.folder)
    # for root, dirnames, filenames in os.walk(args.folder):
    #     for file in filenames:
    #         print(file)
    #
    # for entry in os.scandir(args.folder):
    #     print(entry.is_file())