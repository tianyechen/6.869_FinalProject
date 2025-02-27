import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from scipy import io
from scipy import sparse
from scipy.sparse import linalg
import colorsys
import os
import time
import sys

# os.system('cls')
# os.system('reset')

def yiq_to_rgb(y, i, q):                                                        # the code from colorsys.yiq_to_rgb is modified to work for arrays
    r = y + 0.948262*i + 0.624013*q
    g = y - 0.276066*i - 0.639810*q
    b = y - 1.105450*i + 1.729860*q
    r[r < 0] = 0
    r[r > 1] = 1
    g[g < 0] = 0
    g[g > 1] = 1
    b[b < 0] = 0
    b[b > 1] = 1
    return (r, g, b)

# ---------------------------------------------------------------------------- #
# ------------------------------- PREPARE ------------------------------------ #
# ---------------------------------------------------------------------------- #
o_img = 'liang_bw_smol.bmp'
m_img = 'liang_pencil_smol.bmp'
lol_img = 'example.bmp'

# o_img = input('Enter original file name (using single quotes): ')
# m_img = input('Enter marked file name (using single quotes): ')

dir_path = os.path.dirname(os.path.realpath(__file__))
original = misc.imread(os.path.join(dir_path, o_img))
marked = misc.imread(os.path.join(dir_path, m_img))
supp = misc.imread(os.path.join(dir_path, lol_img))

##normalize
original = original.astype(float)/255
marked = marked.astype(float)/255
supp = supp.astype(float)/255

print marked.shape
print original.shape
isColored = abs(original - marked).sum(2) > 0.01                                # isColored as colorIm

# plt.imshow(isColored)
# plt.show()

##get lightness channel from the first image
(Y,_,_) = colorsys.rgb_to_yiq(original[:,:,0],original[:,:,1],original[:,:,2])

##get lightness channel from the second image
(Y2,_,_) = colorsys.rgb_to_yiq(supp[:,:,0],supp[:,:,1],supp[:,:,2])

##get color channels from the marked up images
(_,I,Q) = colorsys.rgb_to_yiq(marked[:,:,0],marked[:,:,1],marked[:,:,2])

YUV = np.zeros(original.shape)                                                  # YUV as ntscIm
YUV[:,:,0] = Y
YUV[:,:,1] = I
YUV[:,:,2] = Q

YUV2 = np.zeros(original.shape)                                                  # YUV as ntscIm for second image
YUV2[:,:,0] = Y2
YUV2[:,:,1] = I
YUV2[:,:,2] = Q

'''
max_d = np.floor(np.log(min(YUV.shape[0],YUV.shape[1]))/np.log(2)-2)
iu = np.floor(YUV.shape[0]/(2**(max_d - 1))) * (2**(max_d - 1))
ju = np.floor(YUV.shape[1]/(2**(max_d - 1))) * (2**(max_d - 1))
isColored = isColored[:iu,:ju].copy()
YUV = YUV[:iu,:ju].copy()
'''
                                                                                # ALTERNATIVE :: colorized = abs(getColorExact( colorIm, YUV ));

# ---------------------------------------------------------------------------- #
# ---------------------------- getExactColor --------------------------------- #
# ---------------------------------------------------------------------------- #

                                                                                # YUV as ntscIm
n = YUV.shape[0]                                                                # n = image height
m = YUV.shape[1]                                                                # m = image width
image_size = n*m

indices_matrix = np.arange(image_size).reshape(n,m,order='F').copy()            # indices_matrix as indsM
# print(indices_matrix) ## lit
# literally matrix of indicies, [1, 256, ... ]
#                               [2, 266, ... ]

wd = 1                                                                          # The radius of window around the pixel to assess
nr_of_px_in_wd = (2*wd + 1)**2                                                  # The number of pixels in the window around one pixel
## = 9
max_nr = image_size * nr_of_px_in_wd                                            # Maximal size of pixels to assess for the hole image
                                                                                # (for now include the full window also for the border pixels)
row_inds = np.zeros(max_nr, dtype=np.int64)
col_inds = np.zeros(max_nr, dtype=np.int64)
vals = np.zeros(max_nr)

# ----------------------------- Interation ----------------------------------- #

length = 0                                                                      # length as len
pixel_nr = 0                                                                    # pixel_nr as consts_len
                                                                                # Nr of the current pixel == row index in sparse matrix

# iterate over pixels in the image column by column
for j in range(m): ##width
    for i in range(n): ##height
        # If current pixel is not colored
        if (not isColored[i,j]):
            window_index = 0
            ## 9 values around the pixel                                        # window_index as tlen
            window_vals = np.zeros(nr_of_px_in_wd)                              # window_vals as gvals
            window_vals2 = np.zeros(nr_of_px_in_wd)                              # window_vals as gvals

            # Then iterate over pixels in the window around [i,j] this trims it lol it is why there is length lmao
            for ii in range(max(0, i-wd), min(i+wd+1,n)):
                for jj in range(max(0, j-wd), min(j+wd+1, m)):

                    # Only if current pixel is not [i,j]
                    if (ii != i or jj != j):
                        row_inds[length] = pixel_nr
                        col_inds[length] = indices_matrix[ii,jj] ##omg this converts it into linear index!!
                        window_vals[window_index] = YUV[ii,jj,0]
                        ##populate second windows_vals
                        window_vals2[window_index] = YUV2[ii,jj,0]
                        ##check if it is in the same fature?? the things around the pixel
                        # print(ii, jj), length
                        length += 1
                        window_index += 1

            center = YUV[i,j,0].copy()                                          # t_val as center
            center2 = YUV2[i,j,0].copy()                                          # t_val as center

            window_vals[window_index] = center ##last value is the center
            window_vals2[window_index] = center2 ##last value is the center
            # print center, center2
                                                                                    # calculate variance of the intensities in a window around pixel [i,j]
            variance = np.mean((window_vals[0:window_index+1] - np.mean(window_vals[0:window_index+1]))**2) # variance as c_var
            sigma = variance * .6                                              #csig as sigma

            variance2 = np.mean((window_vals2[0:window_index+1] - np.mean(window_vals2[0:window_index+1]))**2) # variance as c_var
            sigma2 = variance2 * .6

            # Indeed, magic
            mgv = min(( window_vals[0:window_index+1] - center )**2)
            mgv2 = min(( window_vals2[0:window_index+1] - center2 )**2)

            if (sigma < ( -mgv / np.log(0.01 ))):
                sigma = -mgv / np.log(0.01)
            if (sigma < 0.000002):                                              # avoid dividing by 0
                sigma = 0.000002

            if (sigma2 < ( -mgv2 / np.log(0.01 ))):
                sigma2 = -mgv2 / np.log(0.01)
            if (sigma2 < 0.000002):                                              # avoid dividing by 0
                sigma2 = 0.000002

            window_vals[0:window_index] = np.exp( -((window_vals[0:window_index] - center)**2) / sigma )    # use weighting funtion (2)
            window_vals2[0:window_index] = np.exp( -((window_vals2[0:window_index] - center2)**2) / sigma2 )    # use weighting funtion (2)
            # print sigma, sigma2

            og_percen = 0
            new_percen = 1

            # window_valsfinal = window_vals
            window_valsfinal =  og_percen*window_vals + new_percen*window_vals2
            window_valsfinal[0:window_index] = window_valsfinal[0:window_index] / np.sum(window_valsfinal[0:window_index]) # make the weighting function sum up to 1

            # window_vals[0:window_index] = window_vals[0:window_index] / np.sum(window_vals[0:window_index]) # make the weighting function sum up to 1
            vals[length-window_index:length] = -window_valsfinal[0:window_index]
            # vals[length-window_index:length] = -window_vals[0:window_index]

            # print len(-window_vals[0:window_index])

        # END IF NOT COLORED

        # Add the values for the current pixel
        row_inds[length] = pixel_nr
        col_inds[length] = indices_matrix[i,j]
        # value is 1 if it is a colored pixel and after the pixels around it
        vals[length] = 1
        length += 1
        pixel_nr += 1

    # END OF FOR i
# END OF FOR j



# ---------------------------------------------------------------------------- #
# ------------------------ After Iteration Process --------------------------- #
# ---------------------------------------------------------------------------- #

# Trim to variables to the length that does not include overflow from the edges
vals = vals[0:length]
col_inds = col_inds[0:length]
row_inds = row_inds[0:length]
print pixel_nr
print row_inds
# print n, m
# for i in range(len(vals)):
#     print vals[i]
# -----------------------[]-------- Sparseness --------------------------------- #
# sys.exit('LETS NOT SPARSE IT YET')
##this is a sparse matrix?!!?
A = sparse.csr_matrix((vals, (row_inds, col_inds)), (pixel_nr, image_size))
# io.mmwrite(os.path.join(dir_path, 'sparse_matrix'), A)
b = np.zeros((A.shape[0]))

colorized = np.zeros(YUV.shape)                                                 # colorized as nI = resultant colored image
colorized[:,:,0] = YUV2[:,:,0]
# plt.imshow(colorized)

color_copy_for_nonzero = isColored.reshape(image_size,order='F').copy()         # We have to reshape and make a copy of the view of an array for the nonzero() to work like in MATLAB
colored_inds = np.nonzero(color_copy_for_nonzero)
# print colored_inds
                          # colored_inds as lblInds
# print color_copy_for_nonzero.shape

for t in [1,2]:
    curIm = YUV[:,:,t].reshape(image_size,order='F').copy()
    # print curIm.shape
    b[colored_inds] = curIm[colored_inds]
    # print b[colored_inds]
    new_vals = linalg.spsolve(A, b)                                             # new_vals = linalg.lsqr(A, b)[0] # least-squares solution (much slower), slightly different solutions
    # lsqr returns unexpectedly (ndarray,ndarray) tuple, first is correct so:
    # use new_vals[0] for reshape if you use lsqr
    colorized[:,:,t] = new_vals.reshape(n, m, order='F')

# ---------------------------------------------------------------------------- #
# ------------------------------ Back to RGB --------------------------------- #
# ---------------------------------------------------------------------------- #

(R, G, B) = yiq_to_rgb(colorized[:,:,0],colorized[:,:,1],colorized[:,:,2])
colorizedRGB = np.zeros(colorized.shape)
colorizedRGB[:,:,0] = R                                                         # colorizedRGB as colorizedIm
colorizedRGB[:,:,1] = G
colorizedRGB[:,:,2] = B

f2 = plt.figure()
plt.imshow(colorizedRGB)
plt.show()

misc.imsave(os.path.join(dir_path, 'example3_colorized.bmp'), colorizedRGB, format='bmp')
