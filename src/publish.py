#!/usr/bin/env python

# Copyright 2019 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys, os
sys.path.append(os.path.dirname(__file__))

import config

import argparse

from google.cloud import pubsub_v1
import base64

def pub(project_id: str, topic_id: str, img_pth: str) -> None:
    """Publishes a message to a Pub/Sub topic."""
    # Initialize a Publisher client.
    client = pubsub_v1.PublisherClient()
    # Create a fully qualified identifier of form `projects/{project_id}/topics/{topic_id}`
    topic_path = client.topic_path(project_id, topic_id)

    # Data sent to Cloud Pub/Sub must be a bytestring.
    with open(img_pth, "rb") as image_file:
        data = base64.b64encode(image_file.read())

    # When you publish a message, the client returns a future.
    api_future = client.publish(topic_path, data)
    message_id = api_future.result()

    print(f"Image data sent to {topic_path}: {message_id}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="publish image to topic"
    )
    # parser.add_argument("project_id", help="Google Cloud project ID")
    # parser.add_argument("topic_id", help="Pub/Sub topic ID")
    parser.add_argument("img_pth", help="path of image to be classified")

    args = parser.parse_args()

    if config.pub_sub==True:
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=config.google_application_credentials_path
        project_id = 'learn-pub-sub-337913'
        topic_id = 'requests'
        pub(project_id, topic_id, args.img_pth)
    