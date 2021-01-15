FROM public.ecr.aws/lambda/python:3.8

COPY app/ ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt
RUN pip install torch torchvision torchaudio

CMD ["handler.lambda_handler"]