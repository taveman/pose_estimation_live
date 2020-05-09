# This code downloads the coco dataset from Amazon S3 in parallel.
import multiprocessing
import subprocess

import boto3
from botocore import UNSIGNED
from botocore.client import Config


files = ['val2017.zip', 'annotations_trainval2017.zip', 'train2017.zip']
s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))


def download_and_unzip_from_s3(file_name, bucket_name='fast-ai-coco'):
    print(f'Downloading {file_name}')
    s3.download_file(bucket_name, file_name, file_name)
    print(f'Finished downloading {file_name}. Starting to unzip.')
    subprocess.run([f'unzip {file_name}'])
    print(f'Finished unzipping {file_name}')


if __name__ == '__main__':
    num_cpus = multiprocessing.cpu_count()
    with multiprocessing.Pool(num_cpus) as p:
        p.map(download_and_unzip_from_s3, files)

    print('Done transferring all datasets')
