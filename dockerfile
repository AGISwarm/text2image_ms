FROM python:3.11-slim


COPY dist /dist

RUN mkdir -p /root/pip
RUN --mount=type=secret,id=PIP_INDEX_EXTRA_URL,target=/PIP_INDEX_EXTRA_URL \
    python -m pip install --upgrade pip && \
    python -m pip install /dist/*.whl --no-cache-dir \
    --extra-index-url $(cat /PIP_INDEX_EXTRA_URL) \
    --trusted-host $(cat /PIP_INDEX_EXTRA_URL  | awk -F/ '{print $3}' | awk -F@ '{print $2}')

RUN echo $(ls /dist/*.whl | sed 's/.*\///' | sed 's/-.*//') > /PACKAGE_NAME
ENTRYPOINT  python -m $(cat /PACKAGE_NAME)