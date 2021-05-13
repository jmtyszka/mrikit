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


def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Compute FD statistics from SPM realignment parameters")
    parser.add_argument('-i', '--infile', required=True, help="SPM six-column realignment parameter file")

    # Parse command line arguments
    args = parser.parse_args()

    # Read SPM realign parameters
    # Expects 6-column, space-separated values, one row per TR
    # Column order: dx, dy, dz, rx, ry, rz
    # Displacements in mm, rotations in radians

    p = pd.read_csv(args.infile,
                   header=None,
                   delim_whitespace=True,
                   names=["tx_mm", "ty_mm", "tz_mm", "rx_rad", "ry_rad", "rz_rad"])

    # Extract translations and rotations
    tx_mm = p["tx_mm"]
    ty_mm = p["ty_mm"]
    tz_mm = p["tz_mm"]
    rx_rad = p["rx_rad"]
    ry_rad = p["ry_rad"]
    rz_rad = p["rz_rad"]

    # Use the FD definition from Power JD et al Neuroimage 2012;59:2142
    # http://dx.doi.org/10.1016/j.neuroimage.2011.10.018
    #
    #     FDi = |Δdix|+|Δdiy|+|Δdiz|+|Δαi|+|Δβi|+|Δγi|
    #
    # where
    #
    #     Δdix = d(i −1)x−dix
    #
    # and similarly for the other rigid body parameters [dix diy diz αi βi γi].
    # Rotational displacements were converted from degrees to millimeters by calculating displacement on the surface
    # of a sphere of radius 50 mm, which is approximately the mean distance from the cerebral cortex to the center of
    # the head.

    # Backward differences (forward difference array with leading 0)
    drx = np.insert(np.diff(rx_rad), 0, 0)
    dry = np.insert(np.diff(ry_rad), 0, 0)
    drz = np.insert(np.diff(rz_rad), 0, 0)
    dtx = np.insert(np.diff(tx_mm), 0, 0)
    dty = np.insert(np.diff(ty_mm), 0, 0)
    dtz = np.insert(np.diff(tz_mm), 0, 0)

    # Total framewise displacement (Power 2012)
    r_sphere = 50.0  # mm
    FD = (np.abs(dtx) + np.abs(dty) + np.abs(dtz) +
          np.abs(r_sphere * drx) + np.abs(r_sphere * dry) + np.abs(r_sphere * drz))

    # Report FD stats in mm
    print('')
    print('Framewise Displacement Statistics for {}'.format(args.infile))
    print('  Mean   : {:0.3f} mm'.format(np.mean(FD)))
    print('  Median : {:0.3f} mm'.format(np.median(FD)))
    print('  SD     : {:0.3f} mm'.format(np.std(FD)))
    print('  Min    : {:0.3f} mm'.format(np.min(FD)))
    print('  Max    : {:0.3f} mm'.format(np.max(FD)))

    # Add FD to dataframe
    p["FD_mm"] = FD

    # Save dataframe to CSV file
    outfile = args.infile.replace('.txt', '_FD.csv')
    print('* Saving results to {}'.format(os.path.basename(outfile)))
    p.to_csv(outfile, header=True, float_format='%0.6f', index=False)


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()

