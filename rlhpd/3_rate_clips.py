import argparse
import pickle
from pathlib import Path

import numpy as np
import skvideo.io
import streamlit as st

from common import database, utils


def save_vid(pickle_path, video_path, fps):
    print(pickle_path)
    with open(pickle_path, 'rb') as f:
        clip = pickle.load(f)
    imgs = np.array([state['pov'] for state, action, reward, next_state, done, meta in clip])

    video_path.parent.mkdir(parents=True, exist_ok=True)    
    writer = skvideo.io.FFmpegWriter(
        video_path, 
        inputdict={'-r': str(fps)},
        outputdict={'-r': str(fps), '-vcodec': 'libx264'},
        )
    for idx in range(imgs.shape[0]):
        writer.writeFrame(imgs[idx,...])
    writer.close()

class App:
    def __init__(self, db_path, videos_dir, traj_dir, video_fps) -> None:
        st.set_page_config(page_title="Human preferences user interface", page_icon=None, layout='wide')
        st.title("Human preferences user interface")

        # TODO: Cache so Streamlit doesn't run it on every refresh
        self.npy_dir = Path(traj_dir)
        self.videos_dir = Path(videos_dir)
        self.db = database.AnnotationBuffer(db_path)
        self.video_fps = video_fps
        self.load_css("style.css")
        return

    def load_css(self, css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        return

    def run(self):

        instructions = st.container()
        number_left = st.container()
        left, right = st.columns(2)
        equally_good = st.container()
        ask_for_new = st.container()

        with instructions:
            st.write("Instructions:")
            st.write("Pick the video you prefer for now :)")

        with number_left:
            st.write(self.db.return_all_data())
            st.write(f"Trajectory pairs waiting to be rated: {self.db.get_number_of_unrated_pairs()}")

        # check folder for videos
        # videos will names as ids, same as in table
        # do pre-populated database in previous step that doesn't have the choices yet
        # load the pair vids from the database
        left_id, right_id = self.db.get_one_unrated_pair()

        with left:
            st.write(f"Video ID: `{left_id}`")
            vid_path = self.videos_dir / "left.mp4"
            save_vid(self.npy_dir / f"{left_id}.pickle", vid_path, self.video_fps)
            video_file = open(vid_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

        with right:
            st.write(f"Video ID: `{right_id}`")
            vid_path = self.videos_dir / "right.mp4"
            save_vid(self.npy_dir / f"{right_id}.pickle", vid_path, self.video_fps)
            video_file = open(vid_path, 'rb')
            video_bytes = video_file.read()
            st.video(video_bytes)

        with left:
            choose_left = st.button(
                'The left one is better', key = "left")
            if choose_left:
                self.db.rate_traj_pair(left_id, right_id, 1)
                left_id, right_id = self.db.get_one_unrated_pair()
        with right:
            choose_right = st.button(
                'The right one is better', key = "right")
            if choose_right:
                self.db.rate_traj_pair(left_id, right_id, 2)
                left_id, right_id = self.db.get_one_unrated_pair()
        with equally_good:
            equal = st.button(
                'Both are equally good')
            if equal:
                self.db.rate_traj_pair(left_id, right_id, 3)
                left_id, right_id = self.db.get_one_unrated_pair()
        with ask_for_new:
            undecided = st.button(
                'Cannot decide, give me a new one!')
            if undecided:
                self.db.rate_traj_pair(left_id, right_id, 4)
                left_id, right_id = self.db.get_one_unrated_pair()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rate clips in annotation buffer')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    cfg = utils.load_config(options.config_file)

    app = App(
        db_path=cfg.sampler.db_path,
        videos_dir=cfg.rate_ui.videos_dir,
        traj_dir=cfg.sampler.traj_dir,
        video_fps=cfg.rate_ui.video_fps,
    )
    app.run()
