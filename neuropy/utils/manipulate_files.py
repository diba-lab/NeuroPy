"""Code to manipulate files with python if you can't
figure out how to do it with a bash script"""

from pathlib import Path
import re
import os
import shutil
import numpy as np
import ipynbname


def get_record_time_from_pathname(
    folder, ext: str = "avi", time_str: str = "[0-9]{2}_[0-9]{2}_[0-9]{2}"
):
    """Get recording time of all files in folder assuming it lives in a folder with the
    string in time_str"""

    # Make searching for files work
    assert isinstance(ext, str)
    if ext[0] != "*":
        if ext[0] == ".":
            ext = "*" + ext
        else:
            ext = "*." + ext
    dir = Path(folder)
    files = sorted(dir.glob("**/" + ext))
    folder_dates = []

    # Iterate through all parent folders and find the one matching your time_str
    for file in files:
        dir_date = []
        for fold in file.parts:
            match_obj = re.match(time_str, fold)
            if match_obj is not None:
                dir_date.append(match_obj.group(0))
        try:
            assert (
                len(dir_date) == 1
            ), "time_str you entered returns too many or no matches"
        except AssertionError:
            pass
        folder_dates.extend(dir_date)

    if len(files) != len(folder_dates):
        print("Obtained fewer dates than files, double-check")

    return files, folder_dates


def prepend_time_from_folder_to_file(
    folder,
    ext: str = "avi",
    time_str: str = "[0-9]{2}_[0-9]{2}_[0-9]{2}",
    copy=True,
):
    """Prepend the date from a folder to the file.  Will copy files by default unless
    you change copy to False."""

    ## NRK todo: add in user prompt if copy=True entered!!!
    # Get names to prepend
    files, folder_dates = get_record_time_from_pathname(folder, ext, time_str)
    assert len(files) == len(
        folder_dates
    ), "#files does not match #folders, check time_str and try again"

    success = 0
    for file, date in zip(files, folder_dates):
        new_name = file.with_name(f"{date}_{file.name}")
        if copy:
            shutil.copy(str(file), new_name)
        else:
            file.rename(new_name)
        success += 1

    if success == len(files):
        print(f"{success} files copied/renamed successfully!")
    else:
        print(f"Only {success} of {len(files)} files copied! Check results!")


def decrement_ncs_file_num(folder, decr_amount=1):
    """Subtract decr_amount from end of NCS file name"""
    file_num = []
    for file in sorted(Path(dir).glob("*.ncs")):
        file_num.append(int(str(file).split("_")[-1].split(".")[0]))
    file_num = np.array(file_num)

    ids = np.argsort(file_num)


def save_notebook(
    base_dir: str or Path,
    save_name: str or None = None,
    save_prepend: str or None = None,
):
    """Save a jupyter notebook to base directory noted. Make sure you save it beforehand! (Autosave should take care
    of this, but please check.
    :param: base_dir: directory to save to
    :param save_name: name of notebook, if None uses existing name
    :param save_prepend: if not None, save_prepend is added to the beginning of the notebook name"""

    assert isinstance(save_name, (str, type(None)))
    assert isinstance(save_prepend, (str, type(None)))

    # Assemble/get save_name variables
    nb_fname = ipynbname.name() if save_name is None else save_name
    nb_path = Path(base_dir)
    save_prepend = "" if save_prepend is None else save_prepend

    # Create savename and save file
    nb_copy_savename = nb_path / f"{save_prepend}{nb_fname}.ipynb"
    shutil.copy(str(ipynbname.path()), str(nb_copy_savename))

    # Spit it out for confirmation
    print(f"{ipynbname.path()} saved to {nb_copy_savename}")


if __name__ == "__main__":
    folder = "/data/Working/Trace_FC/Recording_Rats/Finn/2022_01_20_training/3_post/Finn/gobears/2022_01_20"
    # prepend_date_from_folder_to_file(folder)
    get_record_time_from_pathname(folder)
