import os
import numpy as np
import DeepPATH_code.preprocessing.tileLoop_deepzoom4 as tile
import DeepPATH_code.preprocessing.SortTiles as sorter
import DeepPATH_code.preprocessing.jpgtoHDF as converter
import sys

# start_dir = r"DeepPATH_code/preprocessing"

def args_process(mode, args_list):
    if mode == 'tile':
        args = np.array(['-s', str(args_list[0])])
        args = np.append(args, ['-e', str(args_list[1])])
        args = np.append(args, ['-B', str(args_list[2])])
        args = np.append(args, ['-j', str(args_list[3])])
        args = np.append(args, ['-o', args_list[4]])
        args = np.append(args, args_list[5])
        args.flatten()
        return list(args)
    elif mode == 'sort':
        args = np.array(['--SourceFolder', str(args_list[0])])
        args = np.append(args, ['--JsonFile', str(args_list[1])])
        args = np.append(args, ['--Magnification', str(args_list[2])])
        args = np.append(args, ['--MagDiffAllowed', str(args_list[3])])
        args = np.append(args, ['--nSplit', args_list[4]])
        args = np.append(args, ['--SortingOption', args_list[5]])
        args = np.append(args, ['--PatientID', args_list[6]])
        args = np.append(args, ['--PercentTest', 15])
        args = np.append(args, ['--PercentValid', 15])
        args.flatten()
        return list(args)
    elif mode == 'h5':
        args = np.array(['--input_path', args_list[0]])
        args = np.append(args, ['--output', 'testsample.h5'])
        args = np.append(args, ['--chunks', 1])
        args = np.append(args, ['--wSize', 224])
        args = np.append(args, ['--mode', 2])
        args = np.append(args, ['--subset', 'combined'])
        args.flatten()
        return list(args)
overlap=input("Please input overlap size")
background=input("Please input background % allowance")
magnification=input("Please input magnification you what to sort")


input_path = './testSample/*/*svs'
output = './tile_result'
json_path = './testSample/'
files=os.listdir(json_path)
for i in files:
    print(os.path.splitext(i)[1])
    if os.path.splitext(i)[1]=='.json':

        json_path=os.path.join(json_path,i)
        break



args = args_process("tile", [224, overlap, background, 4, output, input_path])
tile.tile_process(args)

s_args=args_process("sort",[output,json_path,magnification,0,1,1,12])
sorter.sort_process(s_args)

c_args = args_process("h5", ['./set_0/TCGA-LUAD/'])
converter.convert_to_h5(c_args)


