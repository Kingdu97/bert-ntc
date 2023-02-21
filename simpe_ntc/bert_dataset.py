import torch
from torch.utils.data import Dataset


class TextClassificationCollator():
        # 생성할때 tokenizer 개체를 받아오고, 최고길이 하이퍼파라미터 지정(메모리효율)

    def __init__(self, tokenizer, max_length, with_text=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.with_text = with_text

    def __call__(self, samples):
        texts = [s['text'] for s in samples]
        labels = [s['label'] for s in samples]

        # pretrained class를 상속받은 bert tokenizer.
        # 이거 3년전만 해도 직접 하나하나 짰어야함.
        encoding = self.tokenizer(
            texts,
            padding=True,
            truncation=True,        # maxlength 기준으로 자르기
            return_tensors="pt",    # pytorch 기준으로 return 해줘
            max_length=self.max_length
        )

        return_value = {
            'input_ids': encoding['input_ids'],
            'attention_mask': encoding['attention_mask'],   # 구멍뚫린곳은 noise 동작하면 안되니까
            'labels': torch.tensor(labels, dtype=torch.long), # 토치 longtype으로 tensor 바꿈
        }
        if self.with_text:
            return_value['text'] = texts

        return return_value


class TextClassificationDataset(Dataset):

            # 세가지만 상속받으면 됨, init, length, getitem

    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, item):
        text = str(self.texts[item])
        label = self.labels[item]
            # 데이터 셋을 데이터 로더에 넣어주는데에 있어 return 해주는 

        return {
            'text': text,
            'label': label,
        }
            # 딕셔너리로 리턴함