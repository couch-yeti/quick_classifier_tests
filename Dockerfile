FROM public.ecr.aws/lambda/python:3.8

COPY app/ ${LAMBDA_TASK_ROOT}

RUN pip install -r requirements.txt

CMD ["handler.lambda_handler"]