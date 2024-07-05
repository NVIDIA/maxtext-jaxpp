ARG BASE_IMAGE
FROM $BASE_IMAGE as base

COPY requirements.txt /tmp/requirements.txt
RUN pip install -U pip && pip install --no-cache-dir -U -r /tmp/requirements.txt

COPY --chown=$USER_UID:$USER_GID . maxtext

RUN pip install --no-cache-dir -e '/workdir/maxtext/third_party/jaxpp[dev]' && \
    rm -rf /workdir/jaxpp
