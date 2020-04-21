#!/usr/bin/env python3
"""
Compute framewise displacement (FD) statistics from SPM realignment parameters

AUTHOR
----
Mike Tyszka

PLACE
----
Caltech Brain Imaging Center

MIT License

Copyright (c) 2019 Mike Tyszka

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


import os
import argparse
import pandas as pd
import numpy as np
import transforms3d as t3d
from glob import glob


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Decompose FLIRT affine transform matrices")
    parser.add_argument('-i', '--infiles',
                        required=True,
                        nargs='+',
                        help="List of FLIRT affine transform .mat files")

    # Parse command line arguments
    args = parser.parse_args()

    # Setup dataframe
    df = pd.DataFrame(columns=['Image', 'tx_mm', 'ty_mm', 'tz_mm', 'rx_deg', 'ry_deg', 'rz_deg'])

    # Loop over list of infile items
    for infile in args.infiles:

        # Expand wildcards with glob
        for fname in glob(infile):

            if fname.endswith('.mat'):

                # Load affine transform matrix from .mat file
                affine_tx = np.genfromtxt(fname=fname)

                # Decompose 4x4 affine matrix
                T, R, Z, S = t3d.affines.decompose44(affine_tx)

                # Extract displacement components
                tx_mm, ty_mm, tz_mm = T

                # Decompose R into rotations about z, y and x axes (Tait-Bryan convention)
                # Note return order of rotations (ZYX)
                rz_rad, ry_rad, rx_rad = t3d.taitbryan.mat2euler(R)

                # Convert to degrees
                rx_deg, ry_deg, rz_deg = np.rad2deg([rx_rad, ry_rad, rz_rad])

                # Add to dataframe
                row = {'Image': os.path.basename(fname),
                       'tx_mm': tx_mm, 'ty_mm': ty_mm, 'tz_mm': tz_mm,
                       'rx_deg': rx_deg, 'ry_deg': ry_deg, 'rz_deg': rz_deg}

                df.append(row, ignore_index=True)

            else:

                print('* {} probably not a FLIRT .mat file - skipping'.format(os.path.basename(fname)))

    # Save dataframe to CSV file
    outfile = 'Conte_Head_Rotations.csv'
    print('Saving results to {}'.format(os.path.realpath(outfile)))
    df.to_csv(outfile, header=True, float_format='%0.1f', index=False)


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()