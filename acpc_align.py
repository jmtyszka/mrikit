#!/usr/bin/env python
"""
Rigid body AC-PC align a brain extracted structural image
- Use the MNI152 2009c nonlin asym T1w brain template as a reference
- Rigid body align and spline resample to MNI space

AUTHOR : Mike Tyszka
PLACE  : Caltech
"""

import os.path as op
import ants
from templateflow import api as tf
import argparse

def main():

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='AC-PC align structural image')
    parser.add_argument('-i', '--infile', required=True, help='Unaligned structural image')
    parser.add_argument('-r', '--resolution', type=float, default=1.0, help='Output resolution (mm)')
    args = parser.parse_args()

    anat_fname = args.infile
    assert op.isfile(anat_fname)

    # Download MNI152 2009c nonlin asym T1w head template
    # The affine registration can be done at 2mm since
    # the extracted rigid body transfer is applied at full resolution
    print('Getting TemplateFlow Reference Image')
    ref_fname = tf.get(
        'MNI152NLin2009cAsym',
        resolution=1,
        suffix='T1w',
        desc='brain'
    )

    # Load reference as an AntsImage
    print(f'Loading reference image {ref_fname}')
    ref_ai = ants.image_read(str(ref_fname))

    # Load structural image (whole head)
    print(f'Loading anatomic image {anat_fname}')
    anat_ai = ants.image_read(anat_fname)

    # Rigid body align anatomic to reference image
    print('Starting rigid registration of anatomic to reference image')
    res_dict = ants.registration(
        fixed=ref_ai,
        moving=anat_ai,
        type_of_transform='Rigid'
    )

    # TODO: Implement arbitrary output spatial resolution from CLI

    # Save AC-PC aligned image to same folder as original anatomic image
    anat_acpc_fname = anat_fname.replace('.nii.gz', '_acpc.nii.gz')
    print(f'Saving ACPC aligned image to {anat_acpc_fname}')
    res_dict['warpedmovout'].to_file(anat_acpc_fname)


if "__main__" in __name__:

    main()