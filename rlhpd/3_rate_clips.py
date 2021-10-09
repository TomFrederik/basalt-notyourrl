import argparse
from pathlib import Path

import numpy as np
import streamlit as st
import torch

from common import database, utils, preference_helpers
from common.reward_model import RewardModel


class App:
    def __init__(self, cfg) -> None:
        st.set_page_config(page_title="Human preferences user interface", page_icon=None, layout='wide')
        st.title("Human preferences user interface")
        
        # TODO: Cache so Streamlit doesn't run it on every refresh
        self.videos_dir = Path(cfg.rate_ui.videos_dir)
        self.db = database.AnnotationBuffer(cfg.sampler.db_path)
        self.video_fps = cfg.rate_ui.video_fps
        self.load_css("style.css")
        self.reward_model_path = Path(cfg.reward.best_model_path)
        return

    def load_css(self, css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
        return

    # @st.cache(suppress_st_warning=True, allow_output_mutation=True, max_entries=1)
    def load_model(self, model_path):
        self.model = RewardModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def run(self):

        left_id, right_id, pref = self.db.get_random_rated_tuple()

        with st.sidebar:
            st.write("Instructions:")
            st.write("Pick the video you prefer for now :)")

            st.write(f"Trajectory pairs waiting to be rated: {self.db.get_number_of_unrated_pairs()}")

            evaluate_model = st.checkbox("Evaluate model", value=False)
            if evaluate_model:
                self.load_model(self.reward_model_path)
                st.write(f"Using model: `{self.reward_model_path}`")

            if st.button("Show me another labelled pair"):
                left_id, right_id, pref = self.db.get_random_rated_tuple()
                # Randomly swap images to avoid having some systematic bias to right or left
                if np.random.randint(2) == 0:
                    left_id, right_id = right_id, left_id
                    if pref == 1:
                        pref = 2
                    elif pref == 2:
                        pref = 1

        left, right = st.columns(2)

        with left:
            st.write(f"`{Path(left_id).parent}`")
            st.write(f"`{Path(left_id).name}`")
            vid_path = self.videos_dir / "left.mp4"
            clip_1 = utils.load_clip_from_file(left_id)
            imgs = np.array([state['pov'] for state, action, reward, next_state, done, meta in clip_1])
            utils.save_vid(imgs, vid_path, self.video_fps)
            # st.write(clip[0][0]['equipped_items'])
            # st.write(clip[0][0]['inventory'])
            # st.write(str(clip[0][0]['vec']))
            with open(vid_path, 'rb') as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes)
            if pref == 1:
                st.success("True label: Better")
            else:
                st.info("True label: Worse")
            
        with right:
            st.write(f"`{Path(right_id).parent}`")
            st.write(f"`{Path(right_id).name}`")
            vid_path = self.videos_dir / "right.mp4"
            clip_2 = utils.load_clip_from_file(right_id)
            imgs = np.array([state['pov'] for state, action, reward, next_state, done, meta in clip_2])
            utils.save_vid(imgs, vid_path, self.video_fps)
            with open(vid_path, 'rb') as video_file:
                video_bytes = video_file.read()
            st.video(video_bytes)
            if pref == 2:
                st.success("True label: Better")
            else:
                st.info("True label: Worse")
            
        if evaluate_model:
            imgs_1, vecs_1 = utils.get_frames_and_vec_from_clip(clip_1)
            imgs_2, vecs_2 = utils.get_frames_and_vec_from_clip(clip_2)
            pred_reward_1 = self.model(imgs_1, vecs_1).sum().item()
            pred_reward_2 = self.model(imgs_2, vecs_2).sum().item()

            probs = preference_helpers.predict_pref_probs(
                self.model, imgs_1.unsqueeze(0), imgs_2.unsqueeze(0), vecs_1.unsqueeze(0), vecs_2.unsqueeze(0))
            probs = probs.detach().numpy()
            pred_judgement = tuple(preference_helpers.probs_to_judgements(probs).squeeze())
            info_1 = f"Pred reward: {pred_reward_1:.3f} (P = {probs.squeeze()[0]:.2f} )"
            info_2 = f"Pred reward: {pred_reward_2:.3f} (P = {probs.squeeze()[1]:.2f} )"
            if pred_judgement == (0.5, 0.5):
                with left:
                    st.info(info_1)
                with right:
                    st.info(info_2)
            elif pred_judgement == (1, 0):
                with left:
                    st.success(info_1)
                with right:
                    st.info(info_2)
            else: # pred_judgement == (0, 1)
                with left:
                    st.info(info_1)
                with right:
                    st.success(info_2)
            # st.write(probs)

        # Feedback
        with left:
            if st.button('Left is better'):
                self.db.rate_traj_pair(left_id, right_id, 1)
                left_id, right_id = self.db.get_one_unrated_pair()
        with right:
            if st.button('Right is better'):
                self.db.rate_traj_pair(left_id, right_id, 2)
                left_id, right_id = self.db.get_one_unrated_pair()
        if st.button('Both are equally good'):
            self.db.rate_traj_pair(left_id, right_id, 3)
            left_id, right_id = self.db.get_one_unrated_pair()
        if st.button('Cannot decide, give me a new one!'):
            self.db.rate_traj_pair(left_id, right_id, 4)
            left_id, right_id = self.db.get_one_unrated_pair()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Rate clips in annotation buffer')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    cfg = utils.load_config(options.config_file)

    app = App(cfg)
    app.run()
