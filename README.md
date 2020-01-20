# object_detection_yolov3

# Requirement

- Install:

```markdown
pip install -r requirement.txt
```

# Freeze
- Download weight at [here](https://drive.google.com/drive/folders/1ziV7giodSOUyx4a4yWsjmx1BCXINi07K?usp=sharing)
- Put weight at folder: `data/`

# How to use:
```sh
    $ git clone https://github.com/trung1309vn/person_detection_yolov3.git

    $ cd person_detection_yolov3

    $ python demo_no_deep_sort.py <cam_folder> <video_number_start> <video_number_end> <mask_name> <date>
```

* Note 1: arguments above are specific for video's contain folder and name, change it in main function for your case, also output folder

* Note 2: According to the code, data and result are saved in data_res folder, which contains data folder, "res" type folders for result videos and people counting in every frame of videos, "cor" type folders for coordinate of people location on video frames, "fre" type folders for people bounding box accumulate, "heat" type folders for saving heat map results which are derived from "cor" and "res", use Heatmap.ipynb to derive it