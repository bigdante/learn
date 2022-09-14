import pandas as pd
import random

import spacy
from torchtext.legacy.data import Field
from torchtext.legacy.data import TabularDataset


def data_preprocess(data_path):
    headers = ['input', 'output']
    with open(data_path, "r") as file:
        lines = file.readlines()
        rows = [[line.split("=")[0], line.split("=")[1]] for line in lines]
    random.shuffle(rows)
    train_data = rows[:int(0.7 * len(rows))]
    test_data = rows[int(0.7 * len(rows)):]
    df_train = pd.DataFrame([row for row in train_data], columns=headers)
    df_test = pd.DataFrame([row for row in test_data], columns=headers)
    df_train.to_csv("train.csv", index=False)
    df_test.to_csv("test.csv", index=False)
    # rows = pd.read_csv("./test.csv")
    # for index, r  in rows.iterrows():
    #     print(type(r['input']),type(r['output']))


if __name__ == '__main__':
    data_path = "./train.txt"
    # data_preprocess(data_path)
    spacy_en = spacy.load('en_core_web_sm')

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    TRG = Field(tokenize=tokenize_en,
                init_token='<sos>',
                eos_token='<eos>',
                lower=True)

    # TEXT = Field(sequential=True, tokenize=tokenize, lower=True)
    # LABEL = Field(sequential=False, use_vocab=False)

    # 构建Dataset
    fields = [("input", SRC), ("output", TRG)]
    # 使用splits方法可以为多个数据集直接创建Dataset
    train, test = TabularDataset.splits(
        path='./',
        train='train.csv',
        test='test.csv',
        format='csv',
        skip_header=True,
        fields=fields)
    print(train)
    # test_datafields = [('id', None), ('comment_text', TEXT)]
    #
    # # 直接创建Dataset(不使用splits)
    # test = TabularDataset(
    #     path=r'data_utils\test.csv',
    #     format='csv',
    #     skip_header=True,
    #     fields=test_datafields
    # )
    # print(train.fields)
    # print(train.examples[0].input)
