#!/usr/bin/env python3
"""
Reconstruct conventional T1w image from unified and inversion time MP2RAGE images
- Requires same bias correction for INV1, INV2 and UNI files

Authors
----
Mike Tyszka, Caltech Brain Imaging Center

Dates
----
2018-10-22 JMT From scratch

MIT License

Copyright (c) 2018 Mike Tyszka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

__version__ = '0.1.0'


import sys
import argparse
import numpy as np
import nibabel as nb
from skimage.filters import threshold_otsu, gaussian
from scipy import logical_or


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Reconstruct T1w image from MP2RAGE data')
    parser.add_argument('-i1', '--inv1', help='MP2RAGE INV1 image filename')
    parser.add_argument('-i2', '--inv2', help='MP2RAGE INV2 image filename')
    parser.add_argument('-u', '--unified', help='MP2RAGE UNI image filename')
    parser.add_argument('-o', '--outname', help='Output T1w image filename')

    # Parse command line arguments
    args = parser.parse_args()
    uni_fname = args.unified
    inv1_fname = args.inv1
    inv2_fname = args.inv2
    t1w_fname = args.outname

    print('Loading UNI image (%s)' % uni_fname)
    try:
        uni_nii = nb.load(uni_fname)
        uni = uni_nii.get_data()
    except:
        print('* Problem loading %s - exiting' % uni_fname)
        sys.exit(1)

    print('Loading INV1 image (%s)' % inv1_fname)
    try:
        inv1_nii = nb.load(inv1_fname)
        inv1 = inv1_nii.get_data()
    except:
        print('* Problem loading %s - exiting' % inv1_fname)
        sys.exit(1)

    print('Loading INV2 image (%s)' % inv2_fname)
    try:
        inv2_nii = nb.load(inv2_fname)
        inv2 = inv2_nii.get_data()
    except:
        print('* Problem loading %s - exiting' % inv2_fname)
        sys.exit(1)

    print('')
    print('Starting T1w image recon')

    # Hardwired Otsu threshold scale factor
    otsu_sf = 0.33

    # Otsu threshold INV1 and INV2 images
    inv1_th = threshold_otsu(inv1) * otsu_sf
    print('  INV1 Otsu threshold : %0.1f' % inv1_th)
    inv1_mask = inv1 > inv1_th

    # Otsu threshold INV1 and INV2 images
    inv2_th = threshold_otsu(inv2) * otsu_sf
    print('  INV2 Otsu threshold : %0.1f' % inv2_th)
    inv2_mask = inv2 > inv2_th

    # Combine INV1 and INV2 masks
    print('  Combining INV masks')
    inv12_mask = logical_or(inv1_mask, inv2_mask)

    # Feather combined mask by one pixel (Gaussin blur)
    print('  Feathering mask')
    inv12_mask = gaussian(inv12_mask, 1.0)

    # Multiply UNI image by feathered mask
    print('  Applying mask to UNI image')
    t1w = uni * inv12_mask

    # Save T1w image
    print('')
    print('Saving T1w image to %s' % t1w_fname)
    t1w_nii = nb.Nifti1Image(t1w, uni_nii.affine)
    t1w_nii.to_filename(t1w_fname)


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
