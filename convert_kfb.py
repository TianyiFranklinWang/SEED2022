import os
from concurrent.futures import ThreadPoolExecutor

CONVERTOR = r".\convertor\KFbioConverter.exe"
INPUT_PATH = r".\input\seed_raw\T0"
OUTPUT_PATH = r".\input\seed_processed\T0"
OUTPUT_DATA_TYPE = ".tif"  # or .svs
NUM_WORKERS = 12
LEVEL = 4  # 2 ~ 9


def convert(file_name):
    instruction = f"{CONVERTOR} {os.path.join(INPUT_PATH, file_name)} {os.path.join(OUTPUT_PATH, file_name.split('.')[0] + OUTPUT_DATA_TYPE)} {LEVEL}"
    print(instruction)
    os.system(instruction)


if __name__ == '__main__':
    file_names = os.listdir(INPUT_PATH)
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH)

    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        executor.map(convert, file_names)
