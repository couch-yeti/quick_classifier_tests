from simple_classifier import common
from simpletransformers.classification import ClassificationModel

model_file = "outputs/"
model = ClassificationModel("bert", model_file, use_cuda=False)


@common.newrelic_wrapper
def lambda_handler(event, context=None):

    return model.predict(
        [
            "For years, computer makers have tried to sell PCs built on Arm processors, a power-efficient family that powers smartphones. Compared with models running on x86 chips from Intel and AMD, though, Arm-based PCs have suffered from performance and software compatibility shortcomings. Now Apple's M1 processors, the Apple-designed member of the Arm family that powers new MacBooks, are changing views of Arm PCs. The M1 chips offer not just good battery life, like Qualcomm's Arm chips in some Windows laptops, but also good performance. At the same time, x86 PCs have improved only gradually."
        ]
    )
