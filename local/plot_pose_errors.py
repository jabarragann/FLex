import sys
import os
import matplotlib.pyplot as plt
import json


def plot_errors(folders, datadir):

    colors = ['g', 'r', 'y', 'b']
    x, rpe_trans1, rpe_rot1, rpe_trans2, rpe_rot2 = [], [], [], [], []
    # rpe trans error
    fig, ax = plt.subplots()
    with open(f"{folders[0]}/end_pose_error_raw.txt", "r") as f:
        lines = f.readlines()
    clean_lines = [x.strip() for x in lines]
    for j in range (len(clean_lines)):
        elements = clean_lines[j].split(',')
        x.append(int(elements[0]))
        rpe_trans1.append(float(elements[1]))
        rpe_rot1.append(float(elements[2]))
    with open(f"{folders[1]}/end_pose_error_raw.txt", "r") as f:
        lines = f.readlines()
    clean_lines = [x.strip() for x in lines]
    for j in range (len(clean_lines)):
        elements = clean_lines[j].split(',')
        rpe_trans2.append(float(elements[1]))
        rpe_rot2.append(float(elements[2]))

    # *colors* is sequence of rgba tuples.
    # *linestyle* is a string or dash tuple. Legal string values are
    # solid|dashed|dashdot|dotted.  The dash tuple is (offset, onoffseq) where
    # onoffseq is an even length tuple of on and off ink in points.  If linestyle
    # is omitted, 'solid' is used.
    # See `matplotlib.collections.LineCollection` for more information.
    scene = os.path.split(datadir)[-1]
    ax.set_title('Trans Errors')
    ax.plot(x, rpe_trans1, colors[0]+'o--', linewidth=2, markersize=1, label="FLex")
    ax.plot(x, rpe_trans2, colors[1]+'o--', linewidth=2, markersize=1, label="Robust-Pose")
    ax.legend()
    fig.savefig(os.path.join(scene+'_trans_error_plot.pdf'))
    # rpe rot error
    fig, ax = plt.subplots()
    ax.set_title("Rot Errors")
    ax.plot(x, rpe_rot1, colors[0]+'o--', linewidth=2, markersize=1, label="FLex")
    ax.plot(x, rpe_rot2, colors[1]+'o--', linewidth=2, markersize=1, label="Robust-Pose")
    ax.legend()
    fig.savefig(os.path.join(scene+'_rot_error_plot.pdf'))



if __name__ == "__main__":
    
    folders = []
    # LocalRF
    # = "Poses_Eval/P2_8_1/localrf_poses"
    #file_name = "transforms_p2_8_1.json"
    # FLex
    basedir = "Poses_Eval/24_rev/FLex_poses"
    folders.append(basedir)
    # Robust Pose
    basedir = "Poses_Eval/24_rev/robust_pose_poses"
    folders.append(basedir)

    datadir = "Endoscopic_NeRF/data/StereoMIS/24_rev"

    plot_errors(folders, datadir)