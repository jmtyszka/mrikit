#!/usr/bin/env python3
"""
Compute framewise displacement (FD) statistics from AFNI head motion correction parameters (*.1D file)

AUTHOR
----
Mike Tyszka

PLACE
----
Caltech Brain Imaging Center

MIT License

Copyright (c) 2021 Mike Tyszka

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
    parser = argparse.ArgumentParser(description="Compute FD statistics from AFNI head motion correction parameters")
    parser.add_argument('-i', '--infile', required=True, help="AFNI 3d six-column realignment parameter file")

    # Parse command line arguments
    args = parser.parse_args()

    # AFNI .1D file format from 3dvolreg help output:
    # -1Dfile ename   Save the motion parameters ONLY in file 'ename'.
    #                 The output is in 6 ASCII formatted columns:
    #                     roll pitch yaw dS  dL  dP
    #
    # Units : degrees CCW for rotation, mm for displacement
    #
    # NOTE: roll, pitch, yaw are defined relative to a heading in z ("ship" sailing along magnet bore)
    #        ** roll  = shaking head 'no' left-right
    #        ** pitch = nodding head 'yes' up-down
    #        ** yaw   = wobbling head sideways (ear toward shoulder)
    #
    # So for RAS orientation, the file columns map to:
    # roll, pitch, yaw, dS, dL, dP : Rz, Rx, Ry, dz, dx, dy

    p = pd.read_csv(args.infile,
                   header=None,
                   delim_whitespace=True,
                   names=["rz_deg", "rx_deg", "ry_deg", "dz_mm", "dx_mm", "dy_mm"])

    # Convert degrees to radians
    p["rx_rad"] = p["rx_deg"] * np.pi / 180.0
    p["ry_rad"] = p["ry_deg"] * np.pi / 180.0
    p["rz_rad"] = p["rz_deg"] * np.pi / 180.0

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

    # Absolute backward finite differences for displacements
    ddx = np.abs(np.diff(p["dx_mm"], prepend=p["dx_mm"][0]))
    ddy = np.abs(np.diff(p["dy_mm"], prepend=p["dy_mm"][0]))
    ddz = np.abs(np.diff(p["dz_mm"], prepend=p["dz_mm"][0]))

    # Absolute scaled backward finite differences for rotations
    r_sphere = 50.0  # mm
    drx = r_sphere * np.abs(np.diff(p["rx_rad"], prepend=p["rx_rad"][0]))
    dry = r_sphere * np.abs(np.diff(p["ry_rad"], prepend=p["ry_rad"][0]))
    drz = r_sphere * np.abs(np.diff(p["rz_rad"], prepend=p["rz_rad"][0]))

    # Framewise displacement timeseries
    # Map rotations to displacement on surface of 50 mm radius sphere (l = r theta)
    FD = ddx + ddy + ddz + drx + dry + drz

    # Report FD stats in mm
    print('')
    print('Framewise Displacement Statistics for {}'.format(args.infile))
    print('  Mean   : {:0.3f} mm'.format(np.mean(FD)))
    print('  Median : {:0.3f} mm'.format(np.median(FD)))
    print('  SD     : {:0.3f} mm'.format(np.std(FD)))
    print('  Min    : {:0.3f} mm'.format(np.min(FD)))
    print('  Max    : {:0.3f} mm'.format(np.max(FD)))

    # Save finite differences and FD to dataframe
    p["drx_mm"] = drx
    p["dry_mm"] = dry
    p["drz_mm"] = dry
    p["FD_mm"] = FD

    # Save dataframe to CSV file
    outfile = args.infile.replace('.1D', '_FD.csv')
    print('Saving results to {}'.format(os.path.basename(outfile)))
    p.to_csv(outfile, header=True, float_format='%0.3f', index=False)


# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()

