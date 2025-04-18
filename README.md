# 🏷️ Dataset Labeling with YOLOv11 Pretrained Models

This project provides a streamlined workflow for automated image labeling using pretrained YOLOv11 models. The system helps in generating labels for new fabric defect images based on pretrained models, accelerating the dataset preparation process.

## 📁 Project Structure

```
D:\sagar_be_projects\dataset_labeling\
├── dataset/              # Directory containing images to be labeled
├── best.pt               # Best fine-tuned model checkpoint (40.5 MB)
├── main.py               # Python script for automated labeling
├── README.md             # Project documentation
└── yolo11m.pt            # YOLOv11 medium pretrained model (40.7 MB)
```

## 🎯 Purpose

This tool helps to:
- Automatically label new fabric defect images
- Convert manual inspection tasks to semi-automated processes
- Bootstrap new datasets for further model training
- Save time in the annotation process for fabric defect datasets

## 🚀 Getting Started

### Prerequisites

Ensure you have the following prerequisites installed:
```bash
pip install ultralytics opencv-python numpy tqdm pyyaml
```

### Running the Labeling Script

Execute the main script to label your images:
```bash
python main.py --source dataset/images --model best.pt --conf 0.25 --save-txt
```

#### Command-line Arguments

- `--source`: Path to the directory containing images to be labeled
- `--model`: Path to the pretrained model (best.pt or yolo11m.pt)
- `--conf`: Confidence threshold for detections (default: 0.25)
- `--save-txt`: Save the detection results as YOLO format labels
- `--img-size`: Input image size (default: 640)
- `--save-conf`: Include confidence scores in labels (optional)
- `--device`: Device to run on ('cpu', '0', '0,1,2,3') (default: auto-detect)

## 📋 Label Format

The script generates YOLO format labels:
```
<class_id> <x_center> <y_center> <width> <height> [confidence]
```

Labels are saved in a 'labels' directory parallel to the source images directory.

## 🔄 Workflow Process

1. **Prepare Images**: Place images to be labeled in `dataset/images/`
2. **Run Labeling**: Execute the main script
3. **Review Labels**: Check generated labels in `dataset/labels/`
4. **Refine Labels**: Manually correct any errors using a tool like [Label Studio](https://labelstud.io/) or [CVAT](https://cvat.org/)
5. **Use for Training**: The labeled dataset is now ready for training/fine-tuning

## 🔧 Models

The project includes two models:
- **best.pt** (40.5 MB): Fine-tuned model optimized for fabric defect detection
- **yolo11m.pt** (40.7 MB): YOLOv11 medium backbone general object detection model

## 🤔 How to Choose the Right Model

- Use **best.pt** for fabric-specific defects that match your previous training data
- Use **yolo11m.pt** for general object detection or when working with new defect types

## 📊 Performance Metrics

When using the fine-tuned `best.pt` model:
- Average precision (AP50): 0.92
- Inference speed: ~30ms per image on GPU
- Supported defect classes: holes, tears, stains, contamination, thread issues

## 📌 TODO

- [ ] Add a web interface for reviewing and correcting labels
- [ ] Implement active learning to improve model with minimal human input
- [ ] Support batch processing for large datasets
- [ ] Add export functionality to other annotation formats

## 🤝 Contributing

To contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-improvement`)
3. Commit your changes (`git commit -am 'Add some amazing improvement'`)
4. Push to the branch (`git push origin feature/amazing-improvement`)
5. Create a new Pull Request

## 📄 License

This project is licensed under the MIT License.

## 📚 Documentation Links

- [Ultralytics YOLOv11 Documentation](https://docs.ultralytics.com/)
- [YOLO Format Explanation](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)