import os
from ocr import prepare_model, process
ROOT_PATH = "/Users/filipgulan/Desktop/test_set/"

model = prepare_model('../keras/weights/weights_ep_253_0.99142.hd5f')

for root, dirs, files in os.walk(ROOT_PATH):
    for filename in files:
        if not filename.endswith("jpg"):
            continue
        print("File:", filename)
        name = filename.split(".")[0]
        file_path = os.path.join(ROOT_PATH, filename)
        output_path = os.path.join("./out/", name + ".txt")
        process(file_path, output_path, model)

