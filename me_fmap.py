#!/usr/bin/env python3
"""
Construct B0 fieldmap from multiecho mag-phase GRE data with TE information

Authors
----
Mike Tyszka, Caltech Brain Imaging Center

Dates
----
2018-07-26 JMT From scratch

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

import nibabel as nb
import numpy as np
from skimage.restoration import unwrap_phase
from scipy.signal import medfilt


def main():

    mag_fname = 'mag.nii.gz'
    phs_fname = 'phs.nii.gz'
    te_fname = 'te.csv'

    print('Loading magnitude images')
    mag_nii = nb.load(mag_fname)
    mag = mag_nii.get_data()

    print('Loading phase images')
    phs_nii = nb.load(phs_fname)
    phs = phs_nii.get_data()

    # TEs in seconds
    print('Loading TEs')
    te = np.genfromtxt(te_fname, delimiter=',')

    # Save affine transform matrix
    T = phs_nii.affine

    # Create signal mask from 10% threshold of first echo magnitude
    mag_0 = mag[:, :, :, 0]
    th = np.max(mag_0) * 0.1
    mask = (mag_0 > th).astype(float)

    # Echo time difference (s)
    dTE = te[1] - te[0]

    # Estimate T2* (exponential model) with mask
    T2star_ms = dTE * 1e3 / (np.log((mag[:, :, :, 0] / mag[:, :, :, 1]))) * mask

    # Scale phase from [-4096, 4096] to [-pi, pi]
    phs = phs * np.pi / 4096.0

    # Phase difference between second and first echoes
    print('phi(TE2) - phi(TE1)')
    dphi = phs[:, :, :, 1] - phs[:, :, :, 0]

    print('Unwrapping phase')
    dphi = unwrap_phase(dphi)

    # 3x3 median filter masked phase difference
    print('Median filtering phase difference')
    dphi = medfilt(dphi, 3) * mask

    # Set median phase difference within mask to 0.0
    print('Setting median phase difference to 0.0 within mask')
    dphi_med = np.median(dphi[mask.astype(bool)])
    dphi -= dphi_med

    # Field offset in rad/s with mask
    dB0_rad_s = dphi / dTE * mask

    # Field offset in Hz with mask
    dB0_Hz = dB0_rad_s / (2.0 * np.pi)

    # Voxel dimensions in m
    dx = np.abs(T[0, 0]) / 1e3
    dy = np.abs(T[1, 1]) / 1e3
    dz = np.abs(T[2, 2]) / 1e3

    # grad(dB0) in mT/m
    print('Field gradient in mT/m')
    gamma_1H = 42.58e3  # Hz/mT
    gx, gy, gz = np.gradient(dB0_Hz, dx, dy, dz)

    # Scale from Hz/m to mT/m and mask
    gx *= mask/gamma_1H
    gy *= mask/gamma_1H
    gz *= mask/gamma_1H

    # Stack field gradient in 4D array
    grad_dB0 = np.stack([gx, gy, gz], axis=3)

    #
    # Save results
    #

    print('Saving dphi')
    dphi_nii = nb.Nifti1Image(dphi, T)
    dphi_nii.to_filename('dphi.nii.gz')

    print('Saving dB0 (rad/s)')
    dB0_rad_s_nii = nb.Nifti1Image(dB0_rad_s, T)
    dB0_rad_s_nii.to_filename('dB0_rad_s.nii.gz')

    print('Saving dB0 (Hz)')
    dB0_Hz_nii = nb.Nifti1Image(dB0_Hz, T)
    dB0_Hz_nii.to_filename('dB0_Hz.nii.gz')

    print('Saving grad(dB0) in mT/m')
    grad_dB0_nii = nb.Nifti1Image(grad_dB0, T)
    grad_dB0_nii.to_filename('grad_dB0.nii.gz')

    print('Saving T2* map')
    T2star_ms_nii = nb.Nifti1Image(T2star_ms, T)
    T2star_ms_nii.to_filename('T2star_ms.nii.gz')

    print('Saving signal mask')
    mask_nii = nb.Nifti1Image(mask, T)
    mask_nii.to_filename('mask.nii.gz')


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
