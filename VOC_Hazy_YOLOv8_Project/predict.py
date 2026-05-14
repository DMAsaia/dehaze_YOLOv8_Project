from ultralytics import YOLO


def main() -> None:
    model = YOLO("best.pt")

    model.predict(
        source="examples/test1.jpg",
        imgsz=512,
        conf=0.25,
        save=True,
    )


if __name__ == "__main__":
    main()
