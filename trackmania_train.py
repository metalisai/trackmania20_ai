import multiprocessing
multiprocessing.set_start_method('spawn', True)

import trackmania
import actor_dqn, actor_sac

from torch.utils.data import Dataset, DataLoader
import random
import uuid
import os
import numpy
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import threading
import argparse
from torch.utils.tensorboard import SummaryWriter
import pickle
import time

from torchrl.data import ListStorage, PrioritizedReplayBuffer

try:
    cpus = multiprocessing.cpu_count()
except NotImplementedError:
    cpus = 6   # arbitrary default

BATCH_SIZE = 32
LEARNING_RATE = 0.001

PICKLE_DATA = False
PICKLE_DIR = "data"
PICKLE_SIZE = 2000

episode_length = 7.0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

img_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(1),
    transforms.ToTensor()
])

img_cache = {}
def collate_gather(batch):
    def cached_img_transform(img):
        global img_cache
        if id(img) in img_cache:
            return img_cache[id(img)]
        else:
            transformed = img_transform(img)
            img_cache[id(img)] = transformed
            return transformed
    states = [torch.tensor(s) for s in batch.state]
    start = time.time()
    ss_imgs = [cached_img_transform(ss) for ss in batch.screenshot]
    actions = [torch.tensor(a) for a in batch.action]
    #rewards = [torch.tensor([r[0] / 5000.0], dtype=torch.float32) for r in batch.reward]
    rewards = [torch.tensor([r[0]], dtype=torch.float32) for r in batch.reward]
    next_states = [torch.tensor(ns) for ns in batch.next_state]
    nss_imgs = [cached_img_transform(nss) for nss in batch.next_screenshot]
    #nss_imgs = pool.map(img_transform, batch.next_screenshot, chunksize=1) # slow af
    dones = [torch.tensor(d) for d in batch.done]
    return states, ss_imgs, actions, rewards, next_states, nss_imgs, dones

def collate_stack(batch):
    states, ss_imgs, actions, rewards, next_states, nss_imgs, dones = batch
    state_batch = torch.stack(states).to(device)
    screenshot_batch = torch.stack(ss_imgs).to(device)
    action_batch = torch.stack(actions).to(device)
    reward_batch = torch.stack(rewards).to(device)

    max_reward = torch.abs(torch.max(reward_batch))
    #if max_reward > 1.0:
        #print(f"max reward should be -1 to 1, but was: {max_reward}")

    next_state_batch = torch.stack(next_states).to(device)
    next_screenshot_batch = torch.stack(nss_imgs).to(device)
    done_batch = torch.stack(dones).to(device)
    return screenshot_batch, state_batch, action_batch, reward_batch, next_screenshot_batch, next_state_batch, done_batch

# collate for torch DataLoader
def collate_fn(batch):
    states = [torch.tensor(b[0]) for b in batch]
    ss_imgs = [img_transform(b[1]) for b in batch]
    actions = [torch.tensor(b[2]) for b in batch]
    next_states = [torch.tensor(b[3]) for b in batch]
    nss_imgs = [img_transform(b[4]) for b in batch]

    rewards = [torch.tensor([b[5][0] / 5000.0]) for b in batch]

    dones = [torch.tensor(b[6]) for b in batch]

    reward_stack = torch.stack(rewards)
    max_reward = torch.abs(torch.max(reward_stack))
    if max_reward > 1.0:
        print(f"max reward should be -1 to 1, but was: {max_reward}")

    return {
        "state": torch.stack(states),
        "screenshot": torch.stack(ss_imgs),
        "action": torch.stack(actions),
        "next_state": torch.stack(next_states),
        "next_screenshot": torch.stack(nss_imgs),
        "reward": reward_stack,
        "done": torch.stack(dones)
    }


def process_recording(ep_memory, wait_for_training=False, skip_count=0):
    global last_loss
    global pickle_data
    global memory
    global writer
    global step

    print("training...")
    losses = []

    def remove_tensors(entry):
        if isinstance(entry, list) and isinstance(entry[0], torch.Tensor):
            return [entry[0].item()]
        else:
            return entry

    '''# add all samples to memory
    for t in range(len(ep_memory)):
        entry = ep_memory.memory[t]
        # if there is a tensor, turn it into normal array
        entry = [remove_tensors(e) for e in entry]

        if isinstance(memory, PrioritizedReplayBuffer):
            memory.add(entry)
        else:
            memory.push(*entry)

        # store data for later if enabled
        if PICKLE_DATA:
            pickle_data.append(ep_memory.memory[t])
            if len(pickle_data) > PICKLE_SIZE:
                guid_name = uuid.uuid4()
                if not os.path.exists(PICKLE_DIR):
                    os.makedirs(PICKLE_DIR)
                with open(f"{PICKLE_DIR}/ep{episode}_{guid_name}.pickle", "wb") as f:
                    pickle.dump(pickle_data, f)
                print(f"pickled {len(pickle_data)} samples")
                pickle_data = []
    '''

    # optimize model
    for t in range(len(ep_memory)):

        entry = ep_memory.memory[t]
        # if there is a tensor, turn it into normal array
        entry = [remove_tensors(e) for e in entry]

        if isinstance(memory, PrioritizedReplayBuffer):
            memory.add(entry)
        else:
            memory.push(*entry)

        # store data for later if enabled
        if PICKLE_DATA:
            pickle_data.append(ep_memory.memory[t])
            if len(pickle_data) > PICKLE_SIZE:
                guid_name = uuid.uuid4()
                if not os.path.exists(PICKLE_DIR):
                    os.makedirs(PICKLE_DIR)
                with open(f"{PICKLE_DIR}/ep{episode}_{guid_name}.pickle", "wb") as f:
                    pickle.dump(pickle_data, f)
                print(f"pickled {len(pickle_data)} samples")
                pickle_data = []

        if skip_count > 0 and t % skip_count != 0:
            continue

        # optimize model
        #if len(memory) < BATCH_SIZE:
        if len(memory) < 1000:
            continue
        #transitions = memory.sample(BATCH_SIZE)
        if isinstance(memory, PrioritizedReplayBuffer):
            sample, info = memory.sample(BATCH_SIZE, return_info=True)
            batch_losses = actor.optimize_model(sample, info["_weight"])
            loss = torch.mean(batch_losses)

            priority = batch_losses.cpu().numpy()
            memory.update_priority(info["index"], priority)
        else:
            transitions, indices, weights = memory.sample_with_priority(BATCH_SIZE)
            #print(f"avg {numpy.average(weights)}")
            #print("i ", indices)
            batch = trackmania.Transition(*zip(*transitions))
            batch = collate_gather(batch)
            batch = collate_stack(batch)

            #loss, bm_loss, cql_loss = optimize_model(batch)
            batch_losses = actor.optimize_model(batch, weights)
            loss = torch.mean(batch_losses)

            priority = batch_losses.cpu().numpy()
            memory.update_priorities(indices, priority) 

        if loss is not None:
            losses.append(loss)
        writer.add_scalar("loss", loss, step)
        #writer.add_scalar("bellman loss", bm_loss, step)
        #writer.add_scalar("cql loss", cql_loss, step)
    if len(losses) > 0:
        avg_loss = sum(losses) / len(losses)
        step += 1
    else:
        avg_loss = 0
    last_loss = avg_loss
    print(f"avg loss {avg_loss}")

    img_cache = {}

    if wait_for_training:
        barr.wait()


def train_online():
    global episode
    global ep_memory
    global last_loss
    global pickle_data
    global barr
    global writer
    global episode_length
    global step

    step = 0

    cap = trackmania.TrackmaniaCapture(time_step=0.1)
    cap.start_state_capture()
    time.sleep(1)

    def select_action(state, screenshot, t):
        img = img_transform(screenshot).unsqueeze(0)
        return actor.select_action(state, img, t)

    ep_memory, total_reward = cap.capture_episode(episode_length, select_action)

    for i_ep in range(10000):
        episode += 1
        actor.set_episode(episode)

        print(f"episode {i_ep}")
        barr.reset()
        threading.Thread(target=process_recording, args=(ep_memory, ), kwargs={'wait_for_training': True}).start()
        ep_memory, total_reward = cap.capture_episode(episode_length, select_action)
        barr.wait()
        actor.update_done()

        writer.add_scalar("episode reward", total_reward, i_ep)
        if last_loss > 0:
            writer.add_scalar("loss", last_loss, episode)
        #img = img_transform(ep_memory.memory[0].screenshot)
        #writer.add_image('image', img, 0)

        if i_ep % 10 == 0:
            actor.save_model(f"model_{episode}.pth")

        if episode > 20:
            episode_length = 10
        if episode > 50:
            episode_length = 15
            randomness = 0.5
        if episode > 100:
            episode_length = 20
            randomness = 0.4
        if episode > 150:
            episode_length = 25
            randomness = 0.3
        if episode > 200:
            episode_length = 30
            randomness = 0.2
        if episode > 250:
            episode_length = 30
            randomness = 0.1

        #cap.memory_to_video(ep_memory, f'episode{episode}.mp4')


def train_from_pickle(pickle_dir):
    global writer
    global last_loss
    global episode
    global step

    step = 0

    files = os.listdir(pickle_dir)
    # filename format: ep{episode}_{uuid}.pickle
    # sort by episode
    files.sort(key=lambda x: int(x.split("_")[0][2:]))
    i = 0
    for file in files:
        data = pickle.load(open(f"{pickle_dir}/{file}", "rb"))
        # note: only used for storage
        mem = trackmania.ReplayMemory(len(data))
        for d in data:
            mem.push(*d)
        print(f"training from {file}...")
        print(data[0])
        process_recording(mem, wait_for_training=False, skip_count=0)
        if last_loss > 0:
            writer.add_scalar("loss (offline)", last_loss, episode)

        episode = 1
        if i > 4:
            actor.update_done()
            #cap.capture_episode(35, select_action)
        i += 1

    actor.save_model(f"model_pickle_{i}.pth")

def preload_pickle_to_buffer(pickle_dir):
    global writer

    files = os.listdir(pickle_dir)
    # filename format: ep{episode}_{uuid}.pickle
    # sort by episode
    files.sort(key=lambda x: int(x.split("_")[0][2:]))
    i = 0
    print("preloading pickles...")
    for file in files:
        data = pickle.load(open(f"{pickle_dir}/{file}", "rb"))
        for d in data:
            if isinstance(memory, PrioritizedReplayBuffer):
                memory.add(d)
            else:
                memory.push(*d)

if __name__ == "__main__":
    #pool = multiprocessing.Pool(1)

    memory = trackmania.ReplayMemory(10000, alpha=0.5, beta=0.4)
    #memory = PrioritizedReplayBuffer(alpha=0.5, beta=0.4, storage=ListStorage(10000), collate_fn=collate_fn, batch_size=BATCH_SIZE)

    barr = threading.Barrier(2)
    last_loss = 0
    pickle_data = []

    args = argparse.ArgumentParser()
    args.add_argument("--load", type=str, default=None)
    args.add_argument("--episode", type=int, default=0)
    args.add_argument("--from_pickle", type=str, default=None)
    args.add_argument("--preload_pickle", type=str, default=None)

    model_path = args.parse_args().load
    episode = args.parse_args().episode
    pickle_dir = args.parse_args().from_pickle
    preload_pickle = args.parse_args().preload_pickle

    actor = actor_dqn.DqnActor(trackmania.state_dim, trackmania.num_actions, 224, lr=LEARNING_RATE, device=device, model_path=model_path)
    #actor = actor_sac.SacActor(trackmania.state_dim, trackmania.num_actions, 224, lr=LEARNING_RATE, device=device, model_path=model_path)

    # tensorboard
    writer = SummaryWriter()

    biased_action_ep = 0
    biased_action = 0


    if preload_pickle is not None:
        preload_pickle_to_buffer(preload_pickle)

    if pickle_dir is not None:
        PICKLE_DATA = False
        train_from_pickle(pickle_dir)
    else:
        train_online()
