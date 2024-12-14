import socket
import struct
import subprocess
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import os
import time
import random
from threading import Thread, Timer, Barrier
import json
import numpy
from pynput.keyboard import Key, Controller
import torch
if os.name == 'posix':
    from Xlib import display, X
elif os.name == 'nt':
    import win32gui
    import win32ui
    import win32con
from torch.utils.data import Dataset
import math

from collections import namedtuple

print("Hellow?")

keyboard = Controller()

DEVNULL = open(os.devnull, 'wb')

HOST = "127.0.0.1"  # The server's hostname or IP address
PORT = 9000  # The port used by the server

WIDTH = 1920
HEIGHT = 1080

OUTPUT_WIDTH = 480
OUTPUT_HEIGHT = 270

USE_FFMPEG_CAPTURE = False

mem = None

last_screenshot = None
last_screenshot_time = 0

num_actions = 9

Transition = namedtuple('Transition', ('state', 'screenshot', 'action', 'next_state', 'next_screenshot', 'reward', 'done'))

state_dim = 8

print("Loading track")
trackInfo = json.load(open('tracks/track3.json', 'r'))

#os.nice(-20)

class ReplayMemory(Dataset):
    def __init__(self, capacity, alpha=0.5, beta=0.4):
        self.capacity = capacity
        self.memory = []

        self.priorities = numpy.zeros(capacity)
        self.normalized_priorities = numpy.zeros(capacity)
        self.normalized_priorities_dirty = True

        self.weights = numpy.zeros(capacity)
        self.position = 0
        self.beta = beta
        self.alpha = alpha

    def calc_weight(self, index):
        if self.normalized_priorities_dirty:
            self.normalize_priorities()
        return 1.0 / (len(self.memory) * self.normalized_priorities[index]) ** self.beta

    def push_many(self, transitions):
        for transition in transitions:
            self.push(*transition, update_weights=False)
        # recalculate weights
        for i in range(len(self.memory)):
            self.weights[i] = self.calc_weight(i)

    def normalize_priorities(self):
        normalized_priorities = self.priorities[:len(self.memory)] / numpy.sum(self.priorities[:len(self.memory)])
        self.normalized_priorities[:len(self.memory)] = normalized_priorities
        sum = numpy.sum(normalized_priorities)
        if abs(sum - 1.0) > 0.0001:
            print(f'error: sum of normalized priorities is {sum}')
        self.normalized_priorities_dirty = False
    
    def push(self, *args, update_weights=True):
        self.priorities[self.position] = 1.0
        self.normalized_priorities_dirty = True
        if len(self.memory) < self.capacity:
            self.memory.append(None) # add new element to the end
            # recalculate weights
            if update_weights:
                for i in range(len(self.memory)):
                    self.weights[i] = self.calc_weight(i)
        self.weights[self.position] = self.calc_weight(self.position)
        self.memory[self.position] = Transition(*args) # overwrite element at position
        self.position = (self.position + 1) % self.capacity # increment position, if position == capacity, position = 0

    def __getitem__(self, index):
        ret = self.memory[index]
        return ret
    
    def sample(self, batch_size):
        #return random.sample(self.memory, batch_size)
        # samples, indices, weights (always 1.0)
        if self.normalized_priorities_dirty:
            self.normalize_priorities()
        # return uniform samples
        indices = random.sample(range(len(self.memory)), batch_size)
        samples = [self.memory[i] for i in indices]
        weights = [self.weights[i] for i in indices]
        return samples, indices, weights

    def sample_with_priority(self, batch_size):
        #sampler = torch.utils.data.sampler.WeightedRandomSampler(self.priorities, batch_size, replacement=False)
        if self.normalized_priorities_dirty:
            self.normalize_priorities()

        indices = random.sample(range(len(self.memory)), batch_size)
        #indices = torch.multinomial(torch.tensor(self.normalized_priorities), batch_size, replacement=False)
        samples = [self.memory[i] for i in indices]

        max_weight = numpy.max(self.weights)
        # normalize weights
        #weights = [w / max_weight for w in self.weights[indices]]
        weights = [self.weights[i] for i in indices]
        #weights = [1.0] * batch_size

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        priorities = numpy.abs(priorities) ** self.alpha
        for i, p in zip(indices, priorities):
            self.priorities[i] = max(p, 1e-5)
            self.normalized_priorities_dirty = True
            self.weights[i] = self.calc_weight(i)
    
    def __len__(self):
        return len(self.memory)

def capture_continuous():
    global last_screenshot
    global last_screenshot_time
    cmd = f"/snap/bin/ffmpeg -nostdin -f x11grab -framerate 60 -video_size {WIDTH}x{HEIGHT} -i :0.0+nomouse -filter:v scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT} -f rawvideo -pix_fmt rgb24 pipe:1" 
    ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE, bufsize=OUTPUT_WIDTH*OUTPUT_HEIGHT*3)
    def read_stderr():
        while True:
            line = ps.stderr.readline()
            text = line.decode('utf-8')
            if line:
                print('FFMPEG STDERR: ', text, end='')
            else:
                break
    Thread(target=read_stderr).start()
    while True:
        data = ps.stdout.read(OUTPUT_WIDTH*OUTPUT_HEIGHT*3)
        if len(data) != OUTPUT_WIDTH*OUTPUT_HEIGHT*3:
            print(f'error: {len(data)}')
            break
        img = Image.frombytes('RGB', (OUTPUT_WIDTH, OUTPUT_HEIGHT), data)
        last_screenshot = img
        #print(f'sc {int(round(time.time() * 1000))}')
        last_screenshot_time = time.time()

def capture_once():
    cmd = f"ffmpeg -f x11grab -video_size {WIDTH}x{HEIGHT} -i :0.0 -filter:v scale={OUTPUT_WIDTH}:{OUTPUT_HEIGHT} -frames:v 1 -f rawvideo -pix_fmt rgb24 pipe:1" 
    ps = subprocess.Popen(cmd,shell=True,stdout=subprocess.PIPE,stderr=DEVNULL)
    data = ps.stdout.read(OUTPUT_WIDTH*OUTPUT_HEIGHT*3)
    img = Image.frombytes('RGB', (OUTPUT_WIDTH, OUTPUT_HEIGHT), data)
    return img

class TrackmaniaCapture:
    def __init__(self, time_step=0.1, frame_stack=2, manual=False):
        self.next_checkpoint = 0
        self.finished = False
        self.barrier = None
        self.reward = 0.0
        self.total_reward = 0.0
        self.time_step = time_step
        self.frame_stack = frame_stack
        self.manual = manual

        self.cur_t = 0.0

        start = trackInfo['Start']
        start = numpy.array([start['X'], start['Y'], start['Z']])
        self.start = start + numpy.array([16.0, 0.0, 16.0])

        finish = trackInfo['Finish']
        finish = numpy.array([finish['X'], finish['Y'], finish['Z']])
        self.finish = finish + numpy.array([16.0, 0.0, 16.0])

        self.reward_distance = 0.0

        if os.name == 'posix':
            dsp = display.Display()
            self.root = dsp.screen().root
        elif os.name == 'nt':
            self.hdesktop = win32gui.GetDesktopWindow()
            self.desktop_dc = win32gui.GetWindowDC(self.hdesktop)
            self.img_dc = win32ui.CreateDCFromHandle(self.desktop_dc)
            self.mem_dc = self.img_dc.CreateCompatibleDC()
            self.screenshot = win32ui.CreateBitmap()
            self.screenshot.CreateCompatibleBitmap(self.img_dc, WIDTH, HEIGHT)
            self.mem_dc.SelectObject(self.screenshot)

        self.min_distance = float('inf')
        self.last_distance = float('inf')
        self.last_speed = 0.0
        self.last_time = time.time()
        self.start_time = time.time()

        self.last_checkpoint_time = time.time()

        self.frame_history = []

        self.checkPoints = []
        for cp in trackInfo['Checkpoints']:
            cp = numpy.array([cp['X'], cp['Y'], cp['Z']])
            cp = cp + numpy.array([16.0, 0.0, 16.0])
            self.checkPoints.append(cp)

        self.checkPoints.append(finish)

    def checkpoint_reached(self):
        base_reward = 30.0 if self.next_checkpoint == len(self.checkPoints)-1 else 10.0
        self.reward += base_reward - 0.2 * self.cur_t if self.next_checkpoint != len(self.checkPoints) else base_reward - 0.5 * self.cur_t
        print(f'checkpoint {self.next_checkpoint} reached reward: {self.reward}')
        self.next_checkpoint += 1
        self.min_distance = float('inf')
        self.last_checkpoint_time = time.time()
        if self.next_checkpoint == len(self.checkPoints):
            self.next_checkpoint = 0
            print("lap completed")
            self.set_state(0)

            self.finished = True

    def update_reward_distance(self, pos):
        self.reward_distance = numpy.linalg.norm(pos - self.checkPoints[self.next_checkpoint-1])

    def capture_state(self):
        self.next_checkpoint = 0

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.connect((HOST, PORT))

            while True:
                data = s.recv(44)
                unpacked = struct.unpack('f'*11, data)
                # speed, distance, pos.x, pos.y, pos.z, steer, gaspedal, isbraking, finished, curGear, engineRpm, 
                #print(f'received {len(data)} speed: {unpacked[0]} gas: {unpacked[6]} brakes: {unpacked[7]} steer: {unpacked[5]} finished: {unpacked[8]}')

                # state - speed, distance, steer, gaspedal, isbraking, curGear, engineRpm
                state = (unpacked[0], unpacked[1], unpacked[5], unpacked[6], unpacked[7], unpacked[9], unpacked[10])
                pos = (unpacked[2], unpacked[3], unpacked[4])

                cur_time = time.time()

                if self.reward_distance == 0.0:
                    self.update_reward_distance(pos)

                if unpacked[8] != 0.0:
                    self.finished = True

                start = self.start
                finish = self.finish

                #print (f'start: {start} pos: {pos}')
                distToStart = ((pos[0]-start[0])**2 + (pos[1]-start[1])**2 + (pos[2]-start[2])**2)**0.5
                distToFinish = ((pos[0]-finish[0])**2 + (pos[1]-finish[1])**2 + (pos[2]-finish[2])**2)**0.5
                #print(f'distance to start: {distToStart}')
                #print(f'distance to finish: {distToFinish}')

                if distToStart < 16.0:
                    self.next_checkpoint = 0

                cps = self.checkPoints
                ni = self.next_checkpoint
                distToNextCheckPoint = ((pos[0]-cps[ni][0])**2 + (pos[1]-cps[ni][1])**2 + (pos[2]-cps[ni][2])**2)**0.5

                # rewards
                if time.time() - self.start_time > 1.0 and not self.finished:
                    # reward for speed
                    if distToNextCheckPoint < self.min_distance and self.next_checkpoint != len(cps)-1:
                        # diff not inf
                        diff = self.min_distance - distToNextCheckPoint
                        if not math.isinf(diff):
                            freward = max(0.0, diff*(unpacked[0]/500.0))
                            freward = min(0.5, freward)
                            self.reward += freward
                        self.min_distance = distToNextCheckPoint

                    # reward for not standing still
                    if unpacked[0] > 5.0:
                        self.reward += 0.002
                    else:
                        self.reward -= 0.001

                    # penalty for crashing
                    speed_dif = unpacked[0] - self.last_speed
                    time_diff = cur_time - self.last_time
                    if time_diff > 0.0:
                        acceleration = speed_dif / time_diff
                    else:
                        acceleration = 0.0
                    if acceleration < -100.0: # braking is around -60
                        cpen = (acceleration*acceleration) / (100000.0)
                        cpen = min(10.0, cpen)
                        self.reward -= cpen
                        print(f'crash penalty: -{cpen}')
                    elif state[4] > 0.1: # small reward for braking
                        brwrd = min(1.0, unpacked[0]/100.0)*0.02
                        #print(f'brake reward: {brwrd}')
                        self.reward += brwrd

                self.last_speed = unpacked[0]

                cpDist = 16.0

                if distToNextCheckPoint < cpDist:
                    self.checkpoint_reached()
                    self.update_reward_distance(pos)

                self.last_distance = distToNextCheckPoint

                #print(f'distance to next checkpoint: {distToNextCheckPoint}')

                self.last_state = state
                self.last_time = cur_time

    def start_state_capture(self):
        thread = Thread(target=self.capture_state)
        thread.start()

    def state_vector_to_index(self, v_state):

        # actions:
        # 0: nothing
        # 1: left
        # 2: right
        # 3: forward
        # 4: brake
        # 5: left + forward
        # 6: right + forward
        # 7: left + brake
        # 8: right + brake

        gas = v_state[3]
        brake = v_state[4]
        steer = v_state[2]

        state = 0
        if gas == 0.0 and brake == 0.0 and steer < 0.0:
            state = 1
        elif gas == 0.0 and brake == 0.0 and steer > 0.0:
            state = 2
        elif gas > 0.0 and brake == 0.0 and steer == 0.0:
            state = 3
        elif gas == 0.0 and brake > 0.0 and steer == 0.0:
            state = 4
        elif gas > 0.0 and brake == 0.0 and steer < 0.0:
            state = 5
        elif gas > 0.0 and brake == 0.0 and steer > 0.0:
            state = 6
        elif gas == 0.0 and brake > 0.0 and steer < 0.0:
            state = 7
        elif gas == 0.0 and brake > 0.0 and steer > 0.0:
            state = 8
        return state

    '''def get_state(self):
        left = keyboard.
        left = keyboard.is_pressed(Key.left)
        right = keyboard.is_pressed(Key.right)
        up = keyboard.is_pressed(Key.up)
        down = keyboard.is_pressed(Key.down)
        if not left and not right and not up and not down:
            return 0
        elif left and not right and not up and not down:
            return 1
        elif not left and right and not up and not down:
            return 2
        elif not left and not right and up and not down:
            return 3
        elif not left and not right and not up and down:
            return 4
        elif left and not right and up and not down:
            return 5
        elif not left and right and up and not down:
            return 6
        elif left and not right and not up and down:
            return 7
        elif not left and right and not up and down:
            return 8'''

    def set_state(self, state):
        if state == 0:
            keyboard.release(Key.left)
            keyboard.release(Key.right)
            keyboard.release(Key.up)
            keyboard.release(Key.down)
        elif state == 1:
            keyboard.press(Key.left)
            keyboard.release(Key.right)
            keyboard.release(Key.up)
            keyboard.release(Key.down)
        elif state == 2:
            keyboard.release(Key.left)
            keyboard.press(Key.right)
            keyboard.release(Key.up)
            keyboard.release(Key.down)
        elif state == 3:
            keyboard.release(Key.left)
            keyboard.release(Key.right)
            keyboard.press(Key.up)
            keyboard.release(Key.down)
        elif state == 4:
            keyboard.release(Key.left)
            keyboard.release(Key.right)
            keyboard.release(Key.up)
            keyboard.press(Key.down)
        elif state == 5:
            keyboard.press(Key.left)
            keyboard.release(Key.right)
            keyboard.press(Key.up)
            keyboard.release(Key.down)
        elif state == 6:
            keyboard.release(Key.left)
            keyboard.press(Key.right)
            keyboard.press(Key.up)
            keyboard.release(Key.down)
        elif state == 7:
            keyboard.press(Key.left)
            keyboard.release(Key.right)
            keyboard.release(Key.up)
            keyboard.press(Key.down)
        elif state == 8:
            keyboard.release(Key.left)
            keyboard.press(Key.right)
            keyboard.release(Key.up)
            keyboard.press(Key.down)

    def capture_screen(self):
        if os.name == "posix":
            raw = self.root.get_image(0, 0, WIDTH, HEIGHT, X.ZPixmap, 0xffffffff)
            image = Image.frombytes("RGB", (WIDTH, HEIGHT), raw.data, "raw", "BGRX")
        elif os.name == "nt":
            self.mem_dc.BitBlt((0, 0), (WIDTH, HEIGHT), self.img_dc, (0, 0), win32con.SRCCOPY)
            bmpinfo = self.screenshot.GetInfo()
            bmpstr = self.screenshot.GetBitmapBits(True)
            image = Image.frombytes('RGB', (bmpinfo['bmWidth'], bmpinfo['bmHeight']), bmpstr, 'raw', 'BGRX')
            
        # downscale, nearest neighbor
        #image = image.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.NEAREST)
        # downscale, bilinear
        #image = image.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.BILINEAR)
        # downscale, bicubic
        image = image.resize((OUTPUT_WIDTH, OUTPUT_HEIGHT), Image.BICUBIC)
        if len(self.frame_history) < self.frame_stack:
            for i in range(self.frame_stack):
                self.frame_history.append(image)
        for i in range(self.frame_stack-1):
            self.frame_history[i] = self.frame_history[i+1]
        self.frame_history[self.frame_stack-1] = image
        return image

    def capture_episode(self, episode_duration, select_action):
        memory = ReplayMemory(10000)

        last_checkpoint_time = time.time()

        global last_screenshot
        global t
        t = 0.0
        # reset
        self.reset()
        # pause
        #pyautogui.press('esc')

        self.barrier = Barrier(2)

        prev_state = self.last_state + (self.cur_t,)
        if USE_FFMPEG_CAPTURE:
            prev_sc = last_screenshot
        else:
            prev_sc = self.capture_screen()

        action = select_action(prev_state, self.frame_history, self.cur_t)
        if not (isinstance(action, list) or isinstance(action, numpy.ndarray)) or not len(action) == 1 or torch.is_tensor(action[0]):
            print("ACTION SHOULD BE LIST WITH ONE ELEMENT")
        if not self.finished:
            self.set_state(action)
        else:
            self.set_state(0)

        timer = None

        i = 0

        self.cur_t = 0.0

        def end_step():
            nonlocal prev_state
            nonlocal prev_sc
            nonlocal timer
            nonlocal i
            nonlocal action
            nonlocal start
            global mem
            global last_screenshot
            global last_screenshot_time
            #print("capture")
            next_state = self.last_state + (self.cur_t,)
            if USE_FFMPEG_CAPTURE:
                next_sc = last_screenshot
            else:
                next_sc = self.capture_screen()

            dif_from_timestep = time.time() - start - self.time_step

            # unix time ms
            #print(f'cap {int(round(time.time() * 1000))}')

            state = self.state_vector_to_index(prev_state)
            finished = self.finished
            rw = self.reward
            memory.push(prev_state, prev_sc, action, next_state, next_sc, [rw], 0.0)
            self.total_reward += rw
            self.reward = 0.0

            #print(f'state: {state} action: {action.item()} reward: {rw} distance: {self.reward_distance} t: {self.cur_t}')

            if USE_FFMPEG_CAPTURE and time.time() - last_screenshot_time > 0.5:
                print(f"no ffmpeg screenshot for {time.time() - last_screenshot_time} seconds")

            '''if mem is not None:
                state = mem.memory[i][0]
                state_idx = self.state_vector_to_index(state)
                self.set_state(state_idx)'''

            prev_state = next_state
            prev_sc = next_sc
            action = select_action(prev_state, self.frame_history, self.cur_t)
            if not (isinstance(action, list) or isinstance(action, numpy.ndarray)) or not len(action) == 1 or torch.is_tensor(action[0]):
                print(f"ACTION SHOULD BE LIST WITH ONE ELEMENT ({action})")
            if not self.manual:
                self.set_state(action)
            else:
                self.set_state(0)

            time_since_last_checkpoint = time.time() - self.last_checkpoint_time

            self.cur_t += self.time_step
            if time_since_last_checkpoint > 15.0:
                print("episode finished by checkpoint timeout")
                memory.push(next_state, next_sc, [0], next_state, next_sc, [-10.0], 1.0)
                self.set_state(0)
                self.barrier.wait()
            elif not finished and self.cur_t < episode_duration:
                start = time.time() - dif_from_timestep
                timer = Timer(self.time_step - dif_from_timestep, end_step)
                timer.start()
            elif finished:
                print("episode finished by finish line")
                memory.push(next_state, next_sc, [0], next_state, next_sc, [20.0], 1.0)
                self.set_state(0)
                self.barrier.wait()
            else:
                print("episode finished by timeout")
                memory.push(next_state, next_sc, [0], next_state, next_sc, [0.0], 1.0)
                keyboard.tap(Key.delete)
                self.set_state(0)
                self.barrier.wait()

            i += 1

        start = time.time()
        timer = Timer(self.time_step, end_step)
        timer.start()

        self.finished = False
        print("waiting end of episode")
        self.barrier.wait()
        print("ep finished, total reward: " + str(self.total_reward))
        timer.cancel()

        return memory, self.total_reward

    def reset(self):
        if not self.finished:
            keyboard.tap(Key.delete)
        self.next_checkpoint = 0
        self.reward = 0.0
        self.last_speed = 0.0
        self.total_reward = 0.0
        self.reward_distance = 0.0
        self.min_distance = float('inf')
        self.last_checkpoint_time = time.time()
        if self.finished:
            # release all keys
            time.sleep(5.0)
            keyboard.tap(Key.enter)
            time.sleep(1.0)
            keyboard.tap(Key.enter)
            time.sleep(0.1)
            keyboard.tap(Key.enter)
            self.start_time = time.time()

    def memory_to_video(self, mem, filename, img_transform=None, output_size=(OUTPUT_WIDTH, OUTPUT_HEIGHT)):
        fps = int(1.0 / self.time_step)
        cmd = f'ffmpeg -f rawvideo -pix_fmt rgb24 -framerate {fps} -s {output_size[0]}x{output_size[1]} -i - -an -vcodec libx264 -preset ultrafast -pix_fmt yuv420p {filename}'
        p = subprocess.Popen(cmd, shell=True, stdin=subprocess.PIPE)
        for i in range(len(mem)):
            state, screenshot, action, next_state, next_screenshot, reward, done = mem.memory[i]
            speed = state[0]
            gas = state[3]
            steer = state[2]
            img3 = Image.new('RGB', (output_size[0], output_size[1]))
            if img_transform is not None:
                screenshot = img_transform(screenshot)
                next_screenshot = img_transform(next_screenshot)
            img3.paste(screenshot, (0,0))
            #img3.paste(next_screenshot, (output_size[0],0))
            d = ImageDraw.Draw(img3)
            d.text((0,0), f'speed: {speed:.2f}', fill=(255,255,255))
            d.text((0,10), f'gas: {gas}', fill=(255,255,255))
            d.text((0,20), f'steer: {steer}', fill=(255,255,255))
            p.stdin.write(img3.tobytes())
        p.stdin.close()
        p.wait()


#capture_once()
#capture_continuous()

if USE_FFMPEG_CAPTURE:
    thread = Thread(target=capture_continuous)
    thread.start()

def playback_memory(mem):
    for i in range(len(mem)):
        state, screenshot, action, next_state, next_screenshot, reward = mem.memory[i]
        print(f'state: {state}')
        print(f'action: {action}')
        print(f'reward: {reward}')
        #print(f'next_state: {next_state}')
        #print(f'next_screenshot: {next_screenshot}')
        #img = Image.frombytes('RGB', (OUTPUT_WIDTH, OUTPUT_HEIGHT), screenshot)
        #img2 = Image.frombytes('RGB', (OUTPUT_WIDTH, OUTPUT_HEIGHT), next_screenshot)
        # concatenate images
        img3 = Image.new('RGB', (OUTPUT_WIDTH*2, OUTPUT_HEIGHT))
        img3.paste(screenshot, (0,0))
        img3.paste(next_screenshot, (OUTPUT_WIDTH,0))
        plt.imshow(img3)
        plt.show(block=False)
        plt.pause(0.001)

def test():
    cap = TrackmaniaCapture(0.05)
    cap.start_state_capture()

    episode_duration = 10.0
    global mem
    time.sleep(5.0)
    print('starting')
    for i in range(5):
        #global mem
        print(f'episode {i}')
        mem = cap.capture_episode(episode_duration, lambda state, sc: torch.tensor([3]))
        #playback_memory(mem)
        memory_to_video(mem, f'episode{i}.mp4')

    exit()

#test()
