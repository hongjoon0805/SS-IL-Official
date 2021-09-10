import data_handler.dataset as data


class DatasetFactory:
    def __init__(self):
        pass

    @staticmethod
    def get_dataset(name):
        if name == 'Imagenet':
            return data.Imagenet()
        elif name == "Google_Landmark_v2_1K":
            return data.Google_Landmark_v2_1K()
        elif name == "Google_Landmark_v2_10K":
            return data.Google_Landmark_v2_10K()
