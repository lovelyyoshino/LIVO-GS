import os

python_path = r"C:\ProgramData\anaconda3\envs\gaussian_splatting\python.exe"
indexs = [10]

for sparse in indexs:
    print(f'======================office4 in {sparse}=====================')
    command = f'{python_path} train.py -s E:\GaussianRecon-test_pose_optimize\GaussianRecon-test_pose_optimize\data\manhattan\\office4 --sparse_num {sparse}'
    os.system(command)
