"""
Module needs to take data from groups and create a single csv file that can be used to 
train Bert classifier.   I'm going to attempt to use Bert for classification, then evaluate 
other setups for long document classification.   

To do this I need to:
1. convert text files to single csv for training 
2. clean data and build dataframes
3. split/ tokenize data 
4. instantiate and train model 
5. run some predictions

"""

from simple_classifier import common
from logzero import logger
import pandas as pd
from sklearn.model_selection import train_test_split
from simpletransformers.classification import ClassificationModel

# raw_df = pd.read_csv(
#     "./simple_classifier/csv_helpers/text_data/data.csv", encoding="latin1"
# )
# raw_df.columns = ["text", "label"]
# raw_df["text"] = raw_df["text"].apply(lambda x: x.replace("\r\n\r\n", " "))

# train, test = train_test_split(
#     raw_df, test_size=0.20, stratify=raw_df["label"]
# )
# print(raw_df.label.unique())

# print(raw_df.head())
if __name__ == "__main__":

    model_file = "outputs/"
    model = ClassificationModel("bert", model_file, num_labels=6)
    # model.train_model(train)

    # result, model_outputs, wrong_predictions = model.eval_model(test)

    predictions, raw_outputs = model.predict(
        [
            """For years, computer makers have tried to sell PCs built on Arm processors, a power-efficient family that powers smartphones. Compared with models running on x86 chips from Intel and AMD, though, Arm-based PCs have suffered from performance and software compatibility shortcomings. Now Apple's M1 processors, the Apple-designed member of the Arm family that powers new MacBooks, are changing views of Arm PCs. The M1 chips offer not just good battery life, like Qualcomm's Arm chips in some Windows laptops, but also good performance. At the same time, x86 PCs have improved only gradually."""
        ]
    )

    print(f"predictions: {predictions}")
