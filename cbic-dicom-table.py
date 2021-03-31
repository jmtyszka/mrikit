#!/usr/bin/env python
"""
Create table of key header tags for all DICOM images in a directory

Authors
----
Mike Tyszka, Caltech Brain Imaging Center

Dates
----
2021-03-31 JMT From scratch

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
import pandas as pd
import pydicom
import argparse
from glob import glob

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Create header tag table from DICOM folder")
    parser.add_argument('-d', '--dicomdir', required=False, default='.', help="DICOM image folder ['.']")

    # Parse command line arguments
    args = parser.parse_args()

    dcm_dir = os.path.realpath(args.dicomdir)
    dcm_list = glob(os.path.join(dcm_dir, '*.dcm'))

    for fpath in dcm_list:

        fname = os.path.basename(fpath)

        ds = pydicom.dcmread(fname)

        print(
            '{:s} | {:s} | {:s} | {:s} | {:d} | {:8.1f} {:8.3f} {:4.1f}'.format(
                fname,
                str(ds.PatientName),
                str(ds.ProtocolName),
                str(ds.AcquisitionDate),
                int(ds.SeriesNumber),
                float(ds.RepetitionTime),
                float(ds.EchoTime),
                float(ds.SliceThickness)
            )
        )

# This is the standard boilerplate that calls the main() function.
if __name__ == '__main__':
    main()
