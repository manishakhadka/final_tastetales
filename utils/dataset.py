import os


def verify_utk(dataset_dir='UTKFace'):
    # Verify if the dataset is available
    if not os.path.exists(dataset_dir):
        print(f"Dataset not found at {dataset_dir}")
        return False
    else:
        print(f"Dataset found at {dataset_dir}")
        return True

verify_utk("datasets/UTKFace/UTKFace")