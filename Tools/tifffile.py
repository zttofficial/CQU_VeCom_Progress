'''
Use the tifffile library to try to convert .npy files into .tif images
'''

import tifffile
from multichannels_splicing import multichannels_splicing

#tifffile functional test
path1 = "D:/CQU/车身对比/车身比对0506-0623_1/3200000100_21042432000601250526_0111_0_.jpg"
path2 = "D:/CQU/车身对比/车身比对0506-0623_1/3200000100_21042432000601250526_0111_0_A.jpg"
count = 0
result,data1,data2 = multichannels_splicing(path1, path2, count)
tifffile.imwrite("D:/CQU/车身对比/npy/test_rgb.tif", result, photometric='rgb')
tifffile.imwrite("D:/CQU/车身对比/npy/test_rgb_compression.tif", result, photometric='rgb', compression='deflate')
#tifffile.imwrite("D:/CQU/车身对比/npy/test_rgb_comp_jpeg.tif", result, photometric='rgb', compression='jpeg')
tifffile.imwrite("D:/CQU/车身对比/npy/test_minisblack.tif", result, photometric='minisblack')
tifffile.imwrite("D:/CQU/车身对比/npy/test_minisblack_compression.tif", result, photometric='minisblack', compression='deflate')
#tifffile.imwrite("D:/CQU/车身对比/npy/test_minisblack_comp_jpeg.tif", result, photometric='minisblack', compression='jpeg')
