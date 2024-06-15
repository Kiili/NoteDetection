import os
import subprocess
import time
import cv2
import numpy as np
import multiprocessing
from tqdm import tqdm


def read_keys_thread(keys, pedal=None):
    try:
        p = subprocess.Popen(["/usr/bin/aseqdump", "-p", "32:0"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        while True:
            data = p.stdout.readline().decode("utf-8")
            data = [a.strip() for a in data.split(" ") if a != ""]
            if pedal is not None and data[1] == "Control" and data[-3] == "64,":
                pedal[0] = data[-1] != "0"
                continue
            if len(data) != 8 or data[1] != "Note":
                continue

            data = {"action": data[2] == "on", "note": int(data[5][:-1]) - 21, "velocity": int(data[7])}
            keys[data["note"]] = data["action"]
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    folder_path = "data5"
    os.makedirs(folder_path, exist_ok=True)
    key_amounts = np.array([0] * 88)
    for file in tqdm(os.listdir(folder_path)):
        if file.split(".")[-1] != "npy":
            continue
        arr = np.load(f"{folder_path}/{file}")
        key_amounts += arr

    # print(key_amounts)
    # exit()

    keys = multiprocessing.Array("b", [False] * 88)
    read_thread = multiprocessing.Process(target=read_keys_thread, args=(keys,))
    read_thread.start()

    vid = cv2.VideoCapture(0)
    vid.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    # os.system(f"rm -rf {folder_path}/*")

    while True:
        vid.grab()
        ret, frame = vid.read()
        k = np.array(keys, dtype=int)

        frame = frame  # cv2.flip(cv2.flip(frame, 0), 1)

        frame = cv2.resize(frame, (640, 480))
        if 1:
            t = round(time.time(), 3)
            cv2.imwrite(f"{folder_path}/{t}.png", frame)
            np.save(f"{folder_path}/{t}", k)
        if 1:
            key_amounts += k
            cv2.imshow('frame', frame)
            #print([str(x) + str(int_to_note(int(x)))[1:3] for x in np.argsort(key_amounts)[:10]], int(max(key_amounts) - np.median(key_amounts)))
            print("\r", " ".join(str(a) for a in key_amounts), end="")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    read_thread.terminate()
