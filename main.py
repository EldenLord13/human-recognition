from ultralytics import YOLO
from collections import Counter
from pathlib import Path

def detect_objects(model: YOLO, image_path: str) -> None:



    print('Run object recognition ...')
    results = model(image_path, verbose =False)[0]

    print(results)
    model = YOLO()

    if results.names and results.boxes is not None:
        labels = results.boxes.cls.tolist()

        label_names = [results.names[int(cls)] for cls in labels]

        counts = Counter(label_names)

        print('Detected objects: ')
        for label, counts in counts.items():
            print(f'{label} : {counts}')
        else:
            print('No objects detected')

        save_path = results.save(filename=f"RESULT_{Path(image_path).stem}.jpeg")
        print(f'RESULTS saved in {save_path}')



def main():
    model = YOLO('yolo11x.pt')
    detect_objects(model, '/Users/dilmurodmaxmudov/Desktop/Dilmurod files/Programming/projects/human recognition/test_photo_2.webp')






if __name__ == '__main__':
    main()