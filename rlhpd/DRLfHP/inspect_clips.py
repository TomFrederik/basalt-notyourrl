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
            # option = st.selectbox(
            #     'Select a trajectory:',
            #     traj_paths)
            # chosen_path = option
            chosen_path = random.choice(traj_paths)
            st.write(chosen_path)
            with open(chosen_path, 'rb') as f:
                clip = pickle.load(f)

            imgs = []
            rewards = []
            for frame in clip:
                img, action, reward = frame
                imgs.append(img)
                rewards.append(reward)
            ImageSequenceClip(list(imgs), fps=20).write_gif('/tmp/left.gif', fps=20)
            st.image('/tmp/left.gif')
            st.metric("Total reward", np.sum(rewards))

        with col2:
            # option = st.selectbox(
            #     'Select a trajectory:',
            #     traj_paths)
            # chosen_path = option
            chosen_path = random.choice(traj_paths)
            st.write(chosen_path)
            with open(chosen_path, 'rb') as f:
                clip = pickle.load(f)

            imgs = []
            rewards = []
            for frame in clip:
                img, action, reward = frame
                imgs.append(img)
                rewards.append(reward)
            ImageSequenceClip(list(imgs), fps=20).write_gif('/tmp/right.gif', fps=20)
            st.image('/tmp/right.gif')
            st.metric("Total reward", np.sum(rewards))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inspect clips')
    parser.add_argument("-d", "--data-dir", default='./output',
                        help="Root directory containing trajectory data. Default: %(default)s")
    options = parser.parse_args()

    app = App()
    app.run(options.data_dir)