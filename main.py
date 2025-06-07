import argparse
from scripts.predict import predict_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--img', type=str, required=True, help='Path to image')
    args = parser.parse_args()

    pred = predict_image(args.img)
    print("Predicted category:", pred)
