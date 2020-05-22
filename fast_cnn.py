import json
from typing import Tuple, List, Dict, TypedDict, Optional

import torch
import itertools
from torch import Tensor
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from scipy.io import loadmat
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from PIL import Image, ImageDraw, JpegImagePlugin, ImageFont


FONT = ImageFont.truetype('/usr/share/fonts/truetype/freefont/FreeSerif.ttf', 12)
utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd_processing_utils')


def build_image_name(image_id: int) -> str:
    return f'{str(image_id).rjust(12, "0")}.jpg'


def get_images_with_person(number_of_images: int = 10) -> List[Optional[int]]:
    with open('./annotations/instances_val2017.json', 'r') as _f:
        instances = json.load(_f)

        for category in instances['categories']:
            if category['supercategory'] == 'person':
                print(f'Category Person with id: {category["id"]}')

        persons_images = []
        for ann in instances['annotations']:
            if number_of_images:
                if ann['category_id'] == 1:
                    persons_images.append(build_image_name(ann['image_id']))
                    number_of_images -= 1
    return persons_images


def draw_key_points(img: Image.Image, key_points: List[Tuple[str, List]]) -> Image.Image:
    drawing = ImageDraw.Draw(img)
    for point in key_points:
        if point[1][2] != 0:
            # drawing.point(xy=[point[1][0], point[1][1]], fill='blue')
            drawing.ellipse(xy=[point[1][0], point[1][1], point[1][0] + 3, point[1][1] + 3], fill='blue')
            drawing.text((point[1][0], point[1][1]), text=point[0], font=FONT)
    return img


def group_by_n_elements(iterable: List, number_of_elements: int) -> List:
    return [iterable[x: x + number_of_elements] for x in range(0, len(iterable), number_of_elements)]


def group_with_body_parts(coordinates: List) -> List:
    body_parts = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
    grouped = group_by_n_elements(iterable=coordinates, number_of_elements=3)
    result = itertools.zip_longest(body_parts, grouped)
    return list(result)


def draw_bbox(img: JpegImagePlugin.JpegImageFile, points: Tuple[float, float, float, float]) -> ImageDraw.Draw:
    """Draws bbox on the image"""
    x_coord, y_coord, width, height = points
    img = img.convert('RGBA')
    overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
    draw = ImageDraw.Draw(overlay)
    draw.rectangle((
        (x_coord, y_coord),
        (x_coord + width, y_coord + height)),
        fill=(128,255,255,90),
        outline=(210,255,255,120),
        width=3
    )
    return Image.alpha_composite(img, overlay)


def crop_bbox(img: JpegImagePlugin.JpegImageFile, points: Tuple[float, float, float, float]) -> ImageDraw.Draw:
    # to_return = img.resize((256, 192), box=(points[0], points[1], points[0] + points[2], points[1] + points[3]))
    to_return = img.resize((256, 192), box=(points[0], points[1], points[2], points[3]))
    return to_return


class DetectedPersons(TypedDict):
    image: Optional[ImageDraw.Draw]
    boxes: List[Tensor]


def detect_person_faster_cnn(path_to_image: str, model) -> DetectedPersons:
    """Detects persons on image and returns image with blist of bounding boxes"""
    data_to_return = {'image': None, 'boxes': [], 'cropped': []}
    with Image.open(path_to_image) as im:

        with torch.no_grad():
            transform = transforms.Compose([transforms.ToTensor(), ])
            tensor_image = transform(im)
            tensor_image = torch.unsqueeze(tensor_image, 0)
            output = model(tensor_image)
            print(output)
            if output:
                output = output[0]

                for idx, label in enumerate(output['labels']):
                    if label.item() == 1 and output['scores'][idx].item() > 0.99:  # if class number is 1
                        data_to_return['boxes'].append(output['boxes'][idx])
                        im = draw_bbox(img=im, points=output['boxes'][idx])
                        cropped_image = crop_bbox(img=im.copy(), points=output['boxes'][idx])
                        data_to_return['cropped'].append(cropped_image)

    data_to_return['image'] = im
    return data_to_return


# def detect_person_ssd(path_to_image: str, model) -> DetectedPersons:
#     """Detects persons on image and returns image with blist of bounding boxes"""
#     data_to_return = {'image': None, 'boxes': [], 'cropped': []}
#     with Image.open(path_to_image) as im:
#
#         with torch.no_grad():
#             transform = transforms.Compose([transforms.ToTensor(), ])
#             tensor_image = transform(im)
#             tensor = utils.prepare_tensor(im, precision == 'fp16')
#
#             tensor_image = torch.unsqueeze(tensor_image, 0)
#             output = model(tensor)
#             results_per_input = utils.decode_results(output)
#             print(results_per_input)
#             best_results_per_input = [utils.pick_best(results, 0.40) for results in results_per_input]
#             print(best_results_per_input)
#
#             data_to_return['image'] = im
#     return data_to_return


if __name__ == '__main__':

    torch.cuda.empty_cache()

    fast_cnn_model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    fast_cnn_model.eval()
    for param in fast_cnn_model.parameters():
        param.requires_grad = False

    # precision = 'fp32'
    # ssd_model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_ssd', model_math=precision)
    #
    # ssd_model.eval()

    fig = plt.figure(figsize=(25, 20))

    image_ids = get_images_with_person(number_of_images=4)

    for idx, image_path in enumerate(image_ids):
        result = detect_person_faster_cnn(path_to_image=f'./val2017/{image_path}', model=fast_cnn_model)
        # result = detect_person_ssd(path_to_image=f'./val2017/{image_path}', model=ssd_model)
        ax = fig.add_subplot(5, 3, idx + 1, xticks=[], yticks=[])
        plt.imshow(result['image'])
        ax.set_title('Person')

    plt.show()
