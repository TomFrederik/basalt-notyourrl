import argparse
import numpy as np
# import plotly.express as px
import pickle
import random
import streamlit as st
import time
from pathlib import Path

import cv2
import minerl
from moviepy.editor import ImageSequenceClip

import utils

class App:
    def __init__(self):
        pass

    @st.cache(suppress_st_warning=True, allow_output_mutation=True, max_entries=1)
    def get_trajectory_paths(self, data_dir: Path):
        st.warning("Cache miss: `get_trajectory_names` ran")
        traj_paths = sorted([x for x in data_dir.glob("*.pickle")])
        return traj_paths

    def run(self, data_dir):
        st.set_page_config(page_title="Clips Viewer", page_icon=None, layout='wide')
        st.title('Clips Viewer')

        data_dir = Path(data_dir)
        st.write(f"Data dir: `{data_dir}`")

        # Select trajectory
        traj_paths = self.get_trajectory_paths(data_dir)

        col1, col2 = st.columns([1,1])
        with col1:
            chosen_path = random.choice(traj_paths)
            st.write(chosen_path)
            with open(chosen_path, 'rb') as f:
                clip_1 = pickle.load(f)

            imgs = [img for img, action, reward in clip_1]
            reward_1 = np.sum([reward for img, action, reward in clip_1])
            ImageSequenceClip(imgs, fps=20).write_gif('/tmp/left.gif', fps=20)
            st.image('/tmp/left.gif')
            st.metric("Total reward", reward_1)

        with col2:
            chosen_path = random.choice(traj_paths)
            st.write(chosen_path)
            with open(chosen_path, 'rb') as f:
                clip_2 = pickle.load(f)

            imgs = [img for img, action, reward in clip_2]
            reward_2 = np.sum([reward for img, action, reward in clip_2])
            ImageSequenceClip(list(imgs), fps=20).write_gif('/tmp/right.gif', fps=20)
            st.image('/tmp/right.gif')
            st.metric("Total reward", reward_2)
        
        judgement = utils.simulate_judgement(clip_1, clip_2)
        if judgement == (0.5, 0.5):
            st.info(f"It's a tie! {judgement}")
        elif judgement == (1, 0):
            with col1:
                st.success(f"Winner! {judgement}")
        else: # judgement == (0, 1)
            with col2:
                st.success(f"Winner! {judgement}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect clips')
    parser.add_argument("-d", "--data-dir", default='./output',
                        help="Root directory containing trajectory data. Default: %(default)s")
    options = parser.parse_args()

    app = App()
    app.run(options.data_dir)