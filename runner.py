import multiprocessing
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from gather_data import read_keys_thread

from utils import *
from PianoModel import *

from mingus.midi import fluidsynth
from mingus.midi import pyfluidsynth as fs

fluidsynth.midi.fs = fs.Synth(gain=0.8)
fluidsynth.init("SalamanderGrandPiano-V3+20200602.sf2", "pulseaudio")
fluidsynth.main_volume(1, 127)

keys = multiprocessing.Array("b", [False] * 88)
pedal = multiprocessing.Array("b", [False])
read_thread = multiprocessing.Process(target=read_keys_thread, args=(keys, pedal))
read_thread.start()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_last_images = 5

model: torch.nn.Module = PianoModelSmallSelf()
model.load_state_dict(torch.load("models/best_model"))
model = model.to(device)
model.eval()

vid = cv2.VideoCapture(0)
vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

playing_notes = dict()
pedal_notes = list()

try:
    with torch.no_grad():
        last_frames = torch.zeros(size=(1, 1, num_last_images, 480, 640), device=device, dtype=torch.float32)
        i = -1
        while True:
            ret, frame = vid.read()
            if not ret:
                break
            i += 1
            play_threshold = 0.7
            stop_threshold = 0.1

            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame = cv2.flip(cv2.flip(frame, 0), 1)

            image = torch.tensor(data=frame / 255,
                                 dtype=torch.float32,
                                 device=device).unsqueeze(0).unsqueeze(0).unsqueeze(0)

            last_frames = torch.cat((last_frames[:, :, -4:], image), dim=2)

            pred = F.sigmoid(model(last_frames))[0].cpu()


            play_notes = (pred > play_threshold).nonzero(as_tuple=True)[0].numpy()

            for play_note in play_notes:  # play new notes
                if play_note in playing_notes:
                    continue
                prob = pred[play_note]
                note = int_to_note(key=int(play_note),
                                   velocity=int(127 * ((1 - prob) / (1 - play_threshold))))
                playing_notes[play_note] = note
                fluidsynth.play_Note(note)
                pedal_notes.append((note, play_note))

            stop_notes = (pred < stop_threshold).nonzero(as_tuple=True)[0].numpy()
            is_pedal = np.array(pedal, dtype=bool)[0]
            for stop_note in stop_notes:  # stop unpressed notes
                if stop_note in playing_notes:
                    del playing_notes[stop_note]

            if not is_pedal:
                for (note, id) in pedal_notes:
                    if id not in playing_notes:
                        fluidsynth.stop_Note(note)

            gt = np.array(keys, dtype=bool).nonzero()[0]
            print("\rpredicted:", np.array(sorted(playing_notes.keys())), "truth:", gt, end=" ", flush=True)

            if 1:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

except KeyboardInterrupt:
    pass

cv2.destroyAllWindows()
vid.release()
read_thread.terminate()
