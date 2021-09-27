import argparse
import cv2
import minerl
import numpy as np
import plotly.express as px
import random
import os
import streamlit as st
from pathlib import Path
import sqlite3
import skvideo.io
from database import return_all_data, get_number_of_unrated_pairs, get_one_unrated_pair, rate_traj_pair, NoUnratedPair

import time


def run_app(videos_folder=None, database_name=None):
    st.set_page_config(page_title="Human preferences user interface", page_icon=None, layout='wide')
    st.title("Human preferences user interface")

    instructions = st.container()
    number_left = st.container()
    left, right= st.columns(2)
    equally_good = st.container()
    ask_for_new = st.container()

    with instructions:
        st.write("Instructions:")
        st.write("Pick the video you prefer for now :)")



    # check folder for videos
    # videos will names as ids, same as in table
    # do pre-populated database in previous step that doesn't have the choices yet
    # load the pair vids from the database
    # https://towardsdatascience.com/python-has-a-built-in-database-heres-how-to-use-it-47826c10648a
    (left_id,right_id) = get_one_unrated_pair()

    with left:
        choose_left = st.button(
            'The left one is better', key = "left")
        if choose_left:
            rate_traj_pair(left_id, right_id, 1)
            (left_id,right_id) = get_one_unrated_pair()
    with right:
        choose_right = st.button(
            'The right one is better', key = "right")
        if choose_right:
            rate_traj_pair(left_id, right_id, 2)
            (left_id,right_id) = get_one_unrated_pair()
    with equally_good:
        equal = st.button(
            'Both are equally good')
        if equal:
            rate_traj_pair(left_id, right_id, 3)
            (left_id,right_id) = get_one_unrated_pair()
    with ask_for_new:
        undecided = st.button(
            'Cannot decide, give me a new one!')
        if undecided:
            rate_traj_pair(left_id, right_id, 4)
            (left_id,right_id) = get_one_unrated_pair()

    with left:
        st.write(f"Video file: {left_id}.mp4")
        npy_to_vid(left_id)
        left_path = Path(videos_folder, f"{left_id}.mp4")
        video_file = open(left_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

    with right:
        st.write(f"Video file: {right_id}.mp4")
        npy_to_vid(right_id)
        right_path = Path(videos_folder, f"{right_id}.mp4")
        video_file = open(right_path, 'rb')
        video_bytes = video_file.read()
        st.video(video_bytes)

    with number_left:
        conn = sqlite3.connect('trajectories.db') 
        c = conn.cursor()
        st.write(return_all_data())
        st.write(f"Trajectory pairs waiting to be rated: {get_number_of_unrated_pairs()}")

def npy_to_vid(id, video_folder = "tmp", npy_folder="trajectories"):
    outputfile = f"{video_folder}/{id}.mp4"
    writer = skvideo.io.FFmpegWriter(outputfile, outputdict={'-vcodec': 'libx264'})
    npy_array=np.load(f"{npy_folder}/{id}.npy").astype(np.uint8)
    for idx in range(npy_array.shape[0]):
        writer.writeFrame(npy_array[idx,...])
    writer.close()

    

if __name__ == '__main__':
    run_app(videos_folder="tmp")