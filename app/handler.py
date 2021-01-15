from simple_classifier import common
import pandas as pd
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split

model_file = "outputs/"
args = {"overwrite_output_dir": True}
# model = ClassificationModel(
#     "bert", model_file, num_labels=6, use_cuda=False, args=args
# )
model = ClassificationModel("bert", model_file, use_cuda=False)
test_sentence = """For years, computer makers have tried to sell PCs built on Arm processors, a power-efficient family that powers smartphones. Compared with models running on x86 chips from Intel and AMD, though, Arm-based PCs have suffered from performance and software compatibility shortcomings. Now Apple's M1 processors, the Apple-designed member of the Arm family that powers new MacBooks, are changing views of Arm PCs. The M1 chips offer not just good battery life, like Qualcomm's Arm chips in some Windows laptops, but also good performance. At the same time, x86 PCs have improved only gradually."""


def get_train_test_data():
    raw_df = pd.read_csv(
        "./simple_classifier/data/data.csv", encoding="latin1"
    )
    raw_df.columns = ["text", "label"]
    raw_df["text"] = raw_df["text"].apply(lambda x: x.replace("\r\n\r\n", " "))

    train, test = train_test_split(
        raw_df, test_size=0.20, stratify=raw_df["label"]
    )
    return train, test


@common.newrelic_wrapper
def lambda_handler(event, context=None):

    # train, test = get_train_test_data()
    # model.train_model(train)

    prediction, raw = model.predict([test_sentence])
    return {"prediction": prediction, "raw_output": raw}


if __name__ == "__main__":
    print(lambda_handler("whatever"))