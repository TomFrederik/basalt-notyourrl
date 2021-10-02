import argparse
import pickle
import random
from pathlib import Path

import common.preference_helpers as pref
import common.utils as utils
import numpy as np
import streamlit as st
import torch
from common.reward_model import RewardModel
from moviepy.editor import ImageSequenceClip


class App:
    def __init__(self, data_dir, reward_model_dir, videos_dir):
        self.data_dir = Path(data_dir)
        self.reward_model_dir = Path(reward_model_dir)
        self.videos_dir = Path(videos_dir)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

    @st.cache(suppress_st_warning=True, allow_output_mutation=True, max_entries=1)
    def get_trajectory_paths(self, data_dir: str):
        data_dir = Path(data_dir)
        st.warning("Cache miss: `get_trajectory_names` ran")
        traj_paths = sorted([x for x in data_dir.glob("*.pickle")])
        return traj_paths

    # @st.cache(suppress_st_warning=True, allow_output_mutation=True, max_entries=1)
    def load_model(self, model_path):
        self.model = RewardModel()
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def run(self):
        st.set_page_config(page_title="Clips Viewer", page_icon=None, layout='wide')
        st.title('Clips Viewer')

        # Select trajectory
        traj_paths = self.get_trajectory_paths(self.data_dir)

        # # Select model
        # model_subdir = st.selectbox(
        #     'Model:',
        #     sorted([x for x in Path(model_dir).glob("*")])
        #     )
        # model_path = st.selectbox(
        #     'Checkpoint:',
        #     sorted([x for x in Path(model_subdir).glob("*.pt")], reverse=True)
        #     )
        # self.load_model(model_path)

        if st.button("See next pair"):

            with torch.no_grad():
                col1, col2 = st.columns([1,1])
                with col1:
                    chosen_path = random.choice(traj_paths)
                    st.write(chosen_path)
                    with open(chosen_path, 'rb') as f:
                        clip_1 = pickle.load(f)
                    reward_1 = np.sum([reward for state, action, reward, next_state, done, meta in clip_1])
                    imgs_1 = [state['pov'] for state, action, reward, next_state, done, meta in clip_1]
                    ImageSequenceClip(imgs_1, fps=20).write_gif(self.videos_dir / 'left.gif', fps=20)
                    st.image(str(self.videos_dir / 'left.gif'), width=300)

                    # imgs_1 = torch.as_tensor(imgs_1, dtype=torch.float32)
                    # imgs_1 = einops.rearrange(imgs_1, 't h w c -> t c h w') / 255
                    # rewards = self.model(imgs_1)
                    # pred_reward_1 = rewards.sum().item()

                with col2:
                    chosen_path = random.choice(traj_paths)
                    st.write(chosen_path)
                    with open(chosen_path, 'rb') as f:
                        clip_2 = pickle.load(f)
                    reward_2 = np.sum([reward for state, action, reward, next_state, done, meta in clip_2])
                    imgs_2 = [state['pov'] for state, action, reward, next_state, done, meta in clip_2]
                    ImageSequenceClip(list(imgs_2), fps=20).write_gif(self.videos_dir / 'right.gif', fps=20)
                    st.image(str(self.videos_dir / 'right.gif'), width=300)

                    # imgs_2 = torch.as_tensor(imgs_2, dtype=torch.float32)
                    # imgs_2 = einops.rearrange(imgs_2, 't h w c -> t c h w') / 255
                    # rewards = self.model(imgs_2)
                    # pred_reward_2 = rewards.sum().item()
                    
                judgement = pref.simulate_judgement(clip_1, clip_2)
                if judgement == (0.5, 0.5):
                    with col1:
                        st.info(f"Total reward: {reward_1}")
                    with col2:
                        st.info(f"Total reward: {reward_2}")
                elif judgement == (1, 0):
                    with col1:
                        st.success(f"Total reward: {reward_1}")
                    with col2:
                        st.info(f"Total reward: {reward_2}")
                else: # judgement == (0, 1)
                    with col1:
                        st.info(f"Total reward: {reward_1}")
                    with col2:
                        st.success(f"Total reward: {reward_2}")
                
                # probs = pref.predict_pref_probs(self.model, imgs_1.unsqueeze(0), imgs_2.unsqueeze(0))
                # pred_judgement = tuple(pref.probs_to_judgements(probs.detach().numpy()).squeeze())
                # if pred_judgement == (0.5, 0.5):
                #     with col1:
                #         st.info(f"Pred reward: {pred_reward_1}")
                #     with col2:
                #         st.info(f"Pred reward: {pred_reward_2}")
                # elif pred_judgement == (1, 0):
                #     with col1:
                #         st.success(f"Pred reward: {pred_reward_1}")
                #     with col2:
                #         st.info(f"Pred reward: {pred_reward_2}")
                # else: # pred_judgement == (0, 1)
                #     with col1:
                #         st.info(f"Pred reward: {pred_reward_1}")
                #     with col2:
                #         st.success(f"Pred reward: {pred_reward_2}")
                # st.write(probs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='View pairs of clips')
    parser.add_argument("-c", "--config-file", default="config.yaml",
                        help="Initial config file. Default: %(default)s")
    options = parser.parse_args()

    # Params
    cfg = utils.load_config(options.config_file)

    app = App(cfg.clips.dir, cfg.reward_model.path, cfg.pref_ui.videos_dir)
    app.run()
