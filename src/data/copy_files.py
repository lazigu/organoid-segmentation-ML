from glob import glob
from shutil import copyfile

import click
import logging
import os
from natsort import natsorted


@click.command()
@click.argument('copy_from', type=click.Path(exists=True), required=False)
@click.argument('copy_to', type=click.Path(exists=True), required=False)
@click.argument('fnames_dir', type=click.Path(exists=True), required=False)
def main(copy_from, copy_to, fnames_dir):
    """
    Parameters
    ----------
    copy_from : str
        dir from where to copy files
    copy_to : str
        dir where to copy files
    fnames_dir : str
        which files to copy
    """
    logger = logging.getLogger(__name__)

    logger.info(f"Copy from: {copy_from}")
    logger.info(f"Copy to: {copy_to}")
    logger.info(f"Copy only files with names as in: {fnames_dir}")

    fnames = natsorted(glob(os.path.join(fnames_dir, "*.tif")))
    copy_these = [os.path.basename(name) for name in fnames]

    filenames = natsorted(glob(os.path.join(copy_from, "*.tif")))

    for file in filenames:
        basename = os.path.basename(file)
        if basename in copy_these:
            print(f"copying file from {os.path.join(copy_from, basename)} to {(os.path.join(copy_to, basename))}")
            copyfile(os.path.join(copy_from, basename), (os.path.join(copy_to, basename)))


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    main()
