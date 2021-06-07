import distutils.util
import glob
import os
import random
import sys
import time

from datetime import datetime

# assume running from Automator dir
def main():
    num_mazes = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    stack = bool(distutils.util.strtobool(sys.argv[2])) if len(sys.argv) > 2 else False

    start = time.time()

    maze_dir = "../Worlds/"
    image_dir = "/raid/Images/"

    now = datetime.now().strftime("%d-%m-%Y_%H-%M")
    subdir = f"{num_mazes}_mazes_{now}"

    maze_sub_dir = os.path.join(maze_dir, subdir)
    image_sub_dir = os.path.join(image_dir, subdir)

    os.mkdir(maze_sub_dir)
    os.mkdir(image_sub_dir)

    if not stack:
        os.mkdir(os.path.join(image_sub_dir, "straight"))
        os.mkdir(os.path.join(image_sub_dir, "right"))
        os.mkdir(os.path.join(image_sub_dir, "left"))

    # past size 14, MazeGen.py breaks
    min_size, max_size = 8, 14

    for i in range(num_mazes):
        size = random.randint(min_size, max_size)
        new_file = os.path.join(maze_sub_dir, f"maze_{i+1}.txt")
        
        print(f"Creating maze {i+1} with size {size}")
        os.system(f"python3 ../MazeGen/MazeGen.py --width {size} --height {size} --out > {new_file}")
        
        print(f"Generating data from maze {i+1}")
        os.system(f"python3 AutoGen.py {new_file} 0 {image_sub_dir}")
    
    if not stack:
        num_straight = len(os.listdir(os.path.join(image_sub_dir, "straight")))
        num_right = len(os.listdir(os.path.join(image_sub_dir, "right")))
        num_left = len(os.listdir(os.path.join(image_sub_dir, "left")))
    else:
        num_straight = len(glob.glob(os.path.join(image_sub_dir, "*straight*")))
        num_right = len(glob.glob(os.path.join(image_sub_dir, "*right*")))
        num_left = len(glob.glob(os.path.join(image_sub_dir, "*left*")))

    num_total = num_straight + num_right + num_left

    print(f"Number of images straight: {num_straight}")
    print(f"Number of images right: {num_right}")
    print(f"Number of images left: {num_left}")
    print(f"Total images: {num_total}")

    end = time.time()
    seconds = int(end-start)
    min, sec = divmod(seconds, 60)
    print(f"Total time: {min} minutes {sec} seconds")


if __name__ == "__main__":
    main()
